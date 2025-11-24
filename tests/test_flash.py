import pytest
import torch

from flash_attn.kernels.flash_pytorch import (
    AttentionPytorch,
    FlashAttentionPytorch,
)
from flash_attn.kernels.flash_triton import FlashAttentionTriton
from flash_attn.utils import setup_logging

logger = setup_logging("test_flash")

def _make_attn_inputs(
    device=None,
    dtype=torch.float32,
    batch_size=8,
    n_queries=512,
    n_keys=512,
    head_dim=64
):
    # torch.random.manual_seed(42)
    q = torch.randn(batch_size, n_queries, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, n_keys, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, n_keys, head_dim, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(batch_size, n_queries, head_dim, device=device, dtype=dtype)

    return q, k, v, do

@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attn_pytorch_vs_reference(is_causal, device="cuda:0", dtype=torch.bfloat16):
    """Test PyTorch FlashAttention implementation against reference."""
    Q, K, V, dO = _make_attn_inputs(device=device, dtype=dtype)
    
    S_ref, O_ref = AttentionPytorch.apply(Q, K, V, is_causal, True)
    S_flash, O_flash = FlashAttentionPytorch.apply(Q, K, V, is_causal, True)
    
    L_ref = O_ref.grad_fn.saved_tensors[0]
    L_flash = O_flash.grad_fn.saved_tensors[0]
    
    torch.testing.assert_close(S_ref, S_flash, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(L_ref, L_flash, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(O_ref, O_flash, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("batch_size,n_queries,n_keys,head_dim", [
    # Basic cases - powers of 2
    (1, 64, 64, 32),
    (2, 128, 128, 64),
    (4, 256, 256, 128),
    (8, 512, 512, 64),
    
    # Non-power-of-2 sequences (will stress your tile boundaries)
    (1, 100, 100, 64),
    (2, 77, 77, 64),        # CLIP text encoder length
    (1, 50, 50, 32),
    
    # Non-square attention (n_queries != n_keys)
    (1, 256, 512, 64),      # More keys than queries
    (1, 512, 256, 64),      # More queries than keys
    (2, 100, 200, 64),      # Non-power-of-2, non-square
    
    # Edge cases
    (1, 16, 16, 64),        # Single tile
    (1, 15, 15, 64),        # Smaller than tile size
    (4, 2048, 1024, 32),    # Large sequence
])
def test_flash_attn_triton_vs_reference(
    is_causal, 
    batch_size, 
    n_queries, 
    n_keys, 
    head_dim, 
    device="cuda:0", 
    dtype=torch.bfloat16
):
    """Test Triton FlashAttention implementation against reference."""
    Q, K, V, dO = _make_attn_inputs(
        device=device, 
        dtype=dtype,
        batch_size=batch_size,
        n_queries=n_queries,
        n_keys=n_keys,
        head_dim=head_dim
    )
    
    _, O_ref = AttentionPytorch.apply(Q, K, V, is_causal, True)
    O_triton = FlashAttentionTriton.apply(Q, K, V, is_causal)
    
    L_ref = O_ref.grad_fn.saved_tensors[0]
    L_triton = O_triton.grad_fn.saved_tensors[0]
    
    torch.testing.assert_close(O_triton, O_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(L_triton, L_ref, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    Q, K, V, dO = _make_attn_inputs(device="cuda:0", dtype=torch.bfloat16)
    
    # Run all implementations
    S_ref, O_ref = AttentionPytorch.apply(Q, K, V, False, True)
    S_debug, O_debug = FlashAttentionPytorch.apply(Q, K, V, False, True)
    O_triton = FlashAttentionTriton.apply(Q, K, V, False)
    
    # Extract L values
    L_ref = O_ref.grad_fn.saved_tensors[0]
    L_debug = O_debug.grad_fn.saved_tensors[0]
    L_triton = O_triton.grad_fn.saved_tensors[0]
    
    # Log all outputs
    logger.info(f"S_ref: {S_ref}")
    logger.info(f"L_ref: {L_ref}")
    logger.info(f"O_ref: {O_ref}")
    logger.info("-" * 50)
    logger.info(f"S_debug: {S_debug}")
    logger.info(f"L_debug: {L_debug}")
    logger.info(f"O_debug: {O_debug}")
    logger.info("-" * 50)
    logger.info(f"L_triton: {L_triton}")
    logger.info(f"O_triton: {O_triton}")
        
    # Test PyTorch debug vs reference
    logger.info("Testing PyTorch debug implementation...")
    torch.testing.assert_close(S_ref, S_debug, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(L_ref, L_debug, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(O_ref, O_debug, rtol=1e-2, atol=1e-2)
    logger.info("✓ PyTorch debug matches reference")
    
    # Test Triton vs reference
    logger.info("Testing Triton implementation...")
    torch.testing.assert_close(O_triton, O_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(L_triton, L_ref, rtol=1e-2, atol=1e-2)
    logger.info("✓ Triton matches reference")
    
    logger.info("✓ All tests passed!")