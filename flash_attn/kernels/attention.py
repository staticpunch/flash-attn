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
from ..utils import softmax
from ..utils import setup_logging

logger = setup_logging("attention_impl")

def copy_docstring(source_func):
    def wrapper(target_func):
        target_func.__doc__ = source_func.__doc__
        return target_func
    return wrapper

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
        
    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    outputs = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return outputs

@copy_docstring(scaled_dot_product_attention)
@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores (matmul)"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
            
    with nvtx.range("computing attention weights (softmax)"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("computing output values (final matmul)"):
        outputs = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return outputs