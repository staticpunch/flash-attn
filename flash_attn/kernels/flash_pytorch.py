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

logger = setup_logging("flash_torch")

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

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        is_causal: Bool = False,
        return_attn_probs: Bool = False,
    ):
        logger.info("Running Flash Attention 2 Pytorch")
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d = Q.shape[-1]
        scale = 1 / (d ** 0.5)
        Q_TILE_SIZE, K_TILE_SIZE = 16, 32
        Tq, Tk = math.ceil(n_queries / Q_TILE_SIZE), math.ceil(n_keys / K_TILE_SIZE)

        # Reshaping Q, K, V so that each outer for-loop iteration 
        # deals with elements from a single batch index
        original_shape = Q.shape
        Q = rearrange(Q, "... queries d -> (...) queries d")             # (n_programs, n_queries, d)
        K = rearrange(K, "... keys d -> (...) keys d")                   # (n_programs, n_keys, d)
        V = rearrange(V, "... keys d -> (...) keys d")                   # (n_programs, n_keys, d)
        O = torch.zeros(Q.shape, device=Q.device, dtype=Q.dtype)         # (n_programs, n_queries, d)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=Q.dtype)    # (n_programs, n_queries)
        n_programs = Q.shape[0]

        if return_attn_probs:                                            # (n_programs, n_queries, n_keys)
            S = torch.zeros((n_programs, n_queries, n_keys),  device=Q.device, dtype=Q.dtype)               

        for pid in range(n_programs):
            # Extract tensors for current batch element
            q, k, v, o, l = Q[pid], K[pid], V[pid], O[pid], L[pid]  # (n_queries, d), (n_keys, d), (n_keys, d), (n_queries, d), (n_queries,)
            
            for i in range(Tq):
                # Load query tile
                Qi = q[i*Q_TILE_SIZE:(i+1)*Q_TILE_SIZE, :]               # (Q_TILE_SIZE, d)
                Oi = o[i*Q_TILE_SIZE:(i+1)*Q_TILE_SIZE, :]               # (Q_TILE_SIZE, d)
                li = l[i*Q_TILE_SIZE:(i+1)*Q_TILE_SIZE]                  # (Q_TILE_SIZE,)
                mi = torch.full((Q_TILE_SIZE,), -torch.inf, device=Q.device, dtype=Q.dtype)
                                                                         # (Q_TILE_SIZE,) - initialized to -inf
                for j in range(Tk):
                    # Load key and value tiles
                    Kj = k[j*K_TILE_SIZE:(j+1)*K_TILE_SIZE, :]           # (K_TILE_SIZE, d)
                    Vj = v[j*K_TILE_SIZE:(j+1)*K_TILE_SIZE, :]           # (K_TILE_SIZE, d)

                    # Step 1: Compute tile of pre-softmax attention scores
                    # Sij: (Q_TILE_SIZE, K_TILE_SIZE) <- Qi @ Kj.T: (Q_TILE_SIZE, d) @ (d, K_TILE_SIZE)
                    Sij = (Qi @ Kj.T) * scale
                    if return_attn_probs: 
                        S[pid, i*Q_TILE_SIZE:(i+1)*Q_TILE_SIZE, j*K_TILE_SIZE:(j+1)*K_TILE_SIZE] = Sij

                    # Step 2: Update running maximum
                    # mi_next (fp32): (Q_TILE_SIZE,) <- mi (fp32): (Q_TILE_SIZE,), Sij: (Q_TILE_SIZE, K_TILE_SIZE)
                    mi_next = torch.max(mi, Sij.max(dim=-1).values)

                    # Step 3: Compute unnormalized softmax numerator
                    # Pij (fp32): (Q_TILE_SIZE, K_TILE_SIZE) <- Sij: (Q_TILE_SIZE, K_TILE_SIZE), mi_next (fp32): (Q_TILE_SIZE,)
                    Pij = torch.exp(Sij - mi_next[:, None])
                    
                    # Step 4: Update running denominator proxy
                    # li_next (fp32): (Q_TILE_SIZE,) <- mi, mi_next, li: (Q_TILE_SIZE,)
                    li_next = torch.exp(mi - mi_next) * li + Pij.sum(dim=-1)
        
                    # Step 5: Update output accumulator
                    # Oi_next: (Q_TILE_SIZE, d) <- mi, mi_next: (Q_TILE_SIZE,), Oi: (Q_TILE_SIZE, d)
                    #                           <- Pij: (Q_TILE_SIZE, K_TILE_SIZE), Vj: (K_TILE_SIZE, d)
                    Oi_next = torch.exp(mi - mi_next)[:, None] * Oi + Pij @ Vj
                    
                    # Step 6: Update running values
                    mi, li, Oi = mi_next, li_next, Oi_next

                # Normalize output by final denominator
                # Oi = (Q_TILE_SIZE, d) <- Oi / li[:, None]
                Oi = Oi / li[:, None]
                
                # Compute logsumexp for backward pass
                # Li = (Q_TILE_SIZE,) <- mi + log(li)
                Li = mi + torch.log(li)

                # Write tile results back
                o[i*Q_TILE_SIZE:(i+1)*Q_TILE_SIZE, :] = Oi
                l[i*Q_TILE_SIZE:(i+1)*Q_TILE_SIZE] = Li

            # Write batch results back
            L[pid], O[pid] = l, o
        
        # Reshape outputs to original shape
        L = L.view(original_shape[:-1])  # (..., n_queries)
        O = O.view(original_shape)       # (..., n_queries, d)

        outputs = (S, O) if return_attn_probs else O
        ctx.save_for_backward(L, Q, K, V, O)
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

class AttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        is_causal: Bool = False,
        return_attn_probs: Bool = False
    ):
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d = Q.shape[-1]
        scale = 1 / (d ** 0.5)

        # Equation 4
        S = einsum(Q, K, '... q d, ... k d -> ... q k') * scale
        if is_causal:
            S = torch.where(
                torch.arange(n_queries, device=S.device)[None, :, None] >= 
                torch.arange(n_keys, device=S.device)[None, None, :],
                S, -1e6
            )

        # Equation 5
        P = softmax(S, dim=-1)

        # Equation 6
        O = einsum(P, V, '... q k, ... k d -> ... q d')

        # Equation 12
        L = torch.logsumexp(S, dim=-1)
        ctx.save_for_backward(L, Q, K, V, O)

        outputs = (S, O) if return_attn_probs else O
        return outputs
        
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError