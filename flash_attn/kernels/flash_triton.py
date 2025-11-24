from __future__ import annotations

import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import einx

import torch
import torch.nn as nn
from torch import Tensor
import torch.cuda.nvtx as nvtx
from jaxtyping import Float, Bool, Int

import triton
import triton.language as tl

from ..utils import softmax
from ..utils import setup_logging
import lovely_tensors as lt
lt.monkey_patch()

logger = setup_logging("flash_triton")

FLASH_ATTENTION_DOCSTRING = """
TODO: Make this docstring comprehensive and more clear.

FlashAttention-2 forward pass with tiled computation and online softmax.

Overview:
This implementation computes the attention output O = softmax(QK^T)V using tiling
to avoid materializing the full attention matrix S = QK^T in high-bandwidth memory 
(HBM). Each tile of the output is computed independently, enabling memory-efficient 
attention computation.

To avoid reading and writing the attention matrix to/from HBM, we use tiling to 
compute each tile of the output independently. This requires computing tiles of the 
attention probabilities P that are tiled in both dimensions (queries and keys). 
Since softmax(S) requires entire rows of S to compute the denominator, we cannot 
compute P in tiles directly. FlashAttention-2 solves this using online softmax, 
which incrementally computes softmax statistics as we process each key tile.

This algorithm achieves O(1) memory complexity with respect to sequence length by 
never materializing the full attention matrix in HBM. Only tiles that fit in SRAM 
are processed at any given time.

Notation:
    - i: Query tile index (subscript)
    - j: Key tile index (superscript)
    - B_q: Tile size along the query dimension
    - B_k: Tile size along the key dimension
    - d: Hidden dimension (not tiled)
    - S_i^{j}: Attention logit scores for query tile i and key tile j, 
               computed as Q_i @ K_j^T
    - T_k: Total number of key tiles

Running Statistics:
For each query tile i, we maintain row-wise running statistics across key tiles:

    m_i^{j} ∈ R^{B_q}: Running maximum across key tiles
        Definition: Maximum value seen so far across all processed key tiles, used for 
            numerically stable softmax computation by offsetting exponentials.
        * Full formula: m_i^{j} = rowmax(S_i^{≤j})
        * Recursive formula: m_i^{j} = max(m_i^{j-1}, rowmax(S_i^{j}))

    l_i^{j} ∈ R^{B_q}: Running sum of exponentials
        Definition: Accumulator for the softmax denominator, representing the sum of 
            exponentials of attention logit scores offset by the running maximum m_i^{j}.
        * Full formula: l_i^{j} = sum(exp(S_i^{≤j} - rowmax(S_i^{≤j})))
        * Recursive formula: 
            l_i^{j} = exp(m_i^{j-1} - m_i^{j}) * l_i^{j-1}  (rescaled previous sum)
                      + rowsum(exp(S_i^{j} - m_i^{j}))      (contribution from new tile)

    Õ_i^{j} ∈ R^{B_q × d}: Unnormalized accumulated output
        Definition: Running sum of weighted values, not yet normalized by the final 
            softmax denominator.

Final Output:
After processing all T_k key tiles, the final attention output is obtained by 
normalizing the accumulated output Õ_i^{T_k} using the final running sum l_i^{T_k}:
    O_i = diag(l_i^{T_k})^{-1} Õ_i^{T_k}

Example: Two-Tile Computation
For a simple case with 2 key/value tiles (j=1, 2), the online softmax algorithm 
proceeds as follows:

First iteration (j=1):
    m_i^{1} = rowmax(S_i^{1}) ∈ R^{B_q}
    l_i^{1} = rowsum(exp(S_i^{1} - m_i^{1})) ∈ R^{B_q}
    Õ_i^{1} = exp(S_i^{1} - m_i^{1}) V^{1} ∈ R^{B_q × d}

Second iteration (j=2):
    m_i^{2} = max(m_i^{1}, rowmax(S_i^{2})) = m_i
    l_i^{2} = exp(m_i^{1} - m_i^{2}) * l_i^{1} + rowsum(exp(S_i^{2} - m_i^{2}))
            = rowsum(exp(S_i^{1} - m_i)) + rowsum(exp(S_i^{2} - m_i)) = l_i
    
    Õ_i^{2} = diag(exp(m_i^{1} - m_i^{2})) * Õ_i^{1} + exp(S_i^{2} - m_i^{2}) V^{2}
            = exp(S_i^{1} - m_i) V^{1} + exp(S_i^{2} - m_i) V^{2}
    
    O_i = diag(l_i^{2})^{-1} Õ_i^{2}

The final normalization step produces the correct attention output equivalent to 
computing softmax over all key tiles at once.
"""

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        is_causal: Bool = False,
    ):
        """
        NOTES:
        Launch grid:
            Launch grid should be set as (T_q, batch_size), meaning each Triton program 
            instance will load only elements from a single batch index, and only read/write 
            to a single query tile of Q, O, and L.

        Kernel structure:
            The kernel should only have a single loop, which will iterate key tiles 1 ≤ j ≤ T_k.
        
        On-chip buffers:
            The on-chip buffers (O_i, l, m) should have dtype tl.float32. If you're 
            accumulating into an output buffer, use the acc argument 
            (acc = tl.dot(..., acc=acc)).

        Type casting:
            Cast P̃^(j) to the dtype of V^(j) before multiplying them, and cast O_i to 
            the appropriate dtype before writing it to global memory. Casting is done 
            with tensor.to. You can get the dtype of a tensor with tensor.dtype, and 
            the dtype of a block pointer/pointer with *_block_ptr.type.element_ty.
        
        Causal mask:
            Add a flag as the last argument to your `autograd.Function` implementation for 
            causal masking. This should be a boolean flag that when set to `True` enables an 
            index comparison for causal masking. 
            
            Your Triton kernel should have a corresponding additional parameter 
            `is_causal: tl.constexpr` (this is a required type annotation). In Triton, construct 
            appropriate index vectors for queries and keys, and compare them to form a square 
            mask of size $B_q \times B_k$. For elements that are masked out, add the constant 
            value of `-1e6` to the corresponding elements of the attention score matrix $S_i^{(j)}$. 
            
            Make sure to save the mask flag for backward using `ctx.is_causal = is_causal`.
        """
        # logger.info("Running Flash Attention 2 Triton")

        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d = Q.shape[-1]
        scale = 1 / (d ** 0.5)
        Q_TILE_SIZE, K_TILE_SIZE = 16, 32
        
        # Reshaping Q, K, V so that each outer for-loop iteration 
        # deals with elements from a single batch index
        original_shape = Q.shape
        Q = rearrange(Q, "... queries d -> (...) queries d")             # (n_programs, n_queries, d)
        K = rearrange(K, "... keys d -> (...) keys d")                   # (n_programs, n_keys, d)
        V = rearrange(V, "... keys d -> (...) keys d")                   # (n_programs, n_keys, d)
        O = torch.zeros(Q.shape, device=Q.device, dtype=Q.dtype)         # (n_programs, n_queries, d)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=Q.dtype)    # (n_programs, n_queries)
        batch_size = Q.shape[0]

        N_QUERY_TILES = triton.cdiv(n_queries, Q_TILE_SIZE)

        grid = (N_QUERY_TILES, batch_size)
        flash_fwd_kernel[grid](
            Q_ptr=Q, K_ptr=K, V_ptr=V,
            O_ptr=O, L_ptr=L,
            stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
            stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
            stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
            stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
            stride_lb=L.stride(0), stride_lq=L.stride(1),
            N_QUERIES=n_queries, N_KEYS=n_keys,
            scale=scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )

        # Reshape outputs to original shape
        L = L.view(original_shape[:-1])  # (..., n_queries)
        O = O.view(original_shape)       # (..., n_queries, d)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O
        
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_idx = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,           # select the k-th batch index
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0), # select the i-th tile index
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,  # select the k-th batch index
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),                   # no offsets, will iterate all later    
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,  # select the k-th batch index
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),                   # no offsets, will iterate all later    
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,           # select the k-th batch index
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0), # select the i-th tile index
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,           # select the k-th batch index
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),   # select the i-th tile index
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # Construct index vectors
    if is_causal: 
        q_offsets = tl.arange(0, Q_TILE_SIZE) + query_tile_idx * Q_TILE_SIZE
        k_offsets = tl.arange(0, K_TILE_SIZE)

    # Load query tile
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")     # (Q_TILE_SIZE, d)
    Oi = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)                           # (Q_TILE_SIZE, d) 
    mi = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) - float('inf')               # (Q_TILE_SIZE,)
    li = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)                              # (Q_TILE_SIZE,)
    input_dtype = Q_block_ptr.type.element_ty                                   # can be (fp16, bf16, or fp32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load key and value tiles       
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, d)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, d)

        # Computation
        Sij        = tl.dot(Qi, Kj.T) * scale
        if is_causal:
            mask   = q_offsets[:, None] >= (k_offsets[None, :] + j * K_TILE_SIZE)
            Sij    = Sij + tl.where(mask, 0, -1.0e6)
            
        mi_next    = tl.maximum(mi, tl.max(Sij, axis=-1))
        Pij        = tl.exp(Sij - mi_next[:, None])
        li_next    = tl.exp(mi - mi_next) * li + tl.sum(Pij, axis=-1)
        Oi_next    = tl.exp(mi - mi_next)[:, None] * Oi \
                     + tl.dot(Pij.to(input_dtype), Vj) # downcastting
        mi, li, Oi = mi_next, li_next, Oi_next

        # Advance pointers
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
 
    # Normalize output by final denominator (fp32)
    Oi = Oi / li[:, None]
    
    # Compute logsumexp for backward pass (fp32)
    Li = mi + tl.log(li)

    # Write tile results back
    tl.store(O_block_ptr, Oi.to(input_dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li.to(input_dtype), boundary_check=(0,))