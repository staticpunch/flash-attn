import pytest
import triton
import triton.language as tl
import torch
from einops import rearrange, einsum
import einx

@triton.jit
def weighted_sum_fwd(
    # x
    x_ptr, # (N, D)
    x_stride_0, x_stride_1,
    # weight
    weight_ptr, # (D,) 
    weight_stride,
    # output
    output_ptr, # (N,)
    output_stride,
    # shapes
    N, D, 
    TILE_N: tl.constexpr, TILE_D: tl.constexpr
):
    row_tile_idx = tl.program_id(0)
    # tl.device_print("pid", row_tile_idx)
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(N, D),
        strides=(x_stride_0, x_stride_1),
        offsets=(row_tile_idx * TILE_N, 0),
        block_shape=(TILE_N, TILE_D),
        order=(1,0)
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride,),
        offsets=(0,),
        block_shape=(TILE_D,),
        order=(0,)
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(N,),
        strides=(output_stride,),
        offsets=(row_tile_idx * TILE_N,),
        block_shape=(TILE_N,),
        order=(0,)
    )

    output_tile = tl.zeros((TILE_N,), dtype=tl.float32)
    
    for i in range(tl.cdiv(D, TILE_D)):
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight_tile = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        # (TILE_D, )  (TILE_N, TILE_D) -> (TILE_N, )
        output_tile += tl.sum(weight_tile * x_tile, axis=-1)

        # advancing
        x_block_ptr = tl.advance(x_block_ptr, (0, TILE_D))
        weight_block_ptr = tl.advance(weight_block_ptr, (TILE_D,))
        
    tl.store(output_block_ptr, output_tile, boundary_check=(0,))

@triton.jit
def weighted_sum_bwd(
    # x: (N, D)
    x_ptr, x_stride_0, x_stride_1,
    # weight: (D,)
    weight_ptr, weight_stride,
    # outgrad: (N,)
    outgrad_ptr, outgrad_stride,
    # x_grad: (N, D)
    x_grad_ptr,
    # weight_grad_partials: (NUM_BLOCKS, D)
    weight_grad_partials_ptr, 
    weight_grad_partials_stride_0,
    weight_grad_partials_stride_1,
    # shapes
    N, D, NUM_BLOCKS,
    TILE_N: tl.constexpr, TILE_D: tl.constexpr
):
    """
    x_grad: (..., D) - outgrad[:, None] * weight[None, :]
    weight_grad: (D,) -  x.T @ outgrad

                       
    weight_grad_tile = x_tile.T @ outgrad_tile
                      (TILE_D, TILE_N) @ (TILE_N,) -> (TILE_D)
    """
    block_idx = tl.program_id(0)
    
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(N, D),
        strides=(x_stride_0, x_stride_1),
        offsets=(block_idx * TILE_N, 0),
        block_shape=(TILE_N, TILE_D),
        order=(1,0)
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride,),
        offsets=(0,),
        block_shape=(TILE_D,),
        order=(0,)
    )
    # tl.device_print("weight", weight_block_ptr)
    outgrad_block_ptr = tl.make_block_ptr(
        outgrad_ptr,
        shape=(N,),
        strides=(outgrad_stride,),
        offsets=(block_idx * TILE_N,),
        block_shape=(TILE_N,),
        order=(0,)
    )

    x_grad_block_ptr = tl.make_block_ptr(
        x_grad_ptr,
        shape=(N, D),
        strides=(x_stride_0, x_stride_1),
        offsets=(block_idx * TILE_N, 0),
        block_shape=(TILE_N, TILE_D),
        order=(1,0)
    )

    weight_grad_partials_block_ptr = tl.make_block_ptr(
        weight_grad_partials_ptr,
        shape=(NUM_BLOCKS, D),
        strides=(
            weight_grad_partials_stride_0, 
            weight_grad_partials_stride_1
        ),
        offsets=(block_idx, 0),
        block_shape=(1, TILE_D),
        order=(1,0)
    )

    outgrad_tile = tl.load(outgrad_block_ptr, boundary_check=(0,), padding_option="zero")
    for i in range(tl.cdiv(D, TILE_D)):
        # --- read ---
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") 
        weight_tile = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        # --- computation ---
        ## (TILE_N, 1) * (1, TILE_D) -> (TILE_N, TILE_D) 
        x_grad_tile = outgrad_tile[:, None] * weight_tile[None, :]
        ## (TILE_D, TILE_N) @ (TILE_N,) -> (TILE_D,)
        ## due to some weird shape processing of triton, i can't simply use `tl.dot(x_tile.T, outgrad_tile)`,
        ## like only accept 2D tensors, and all dimensions should be greater than tile sizes.
        ## insetad, (1, TILE_N) * (TILE_D, TILE_N) -> (TILE_D, TILE_N) -> (1, TILE_D)
        weight_grad_tile = tl.sum(outgrad_tile[None, :] * x_tile.T, axis=-1)[None, :]

        # --- write ---
        tl.store(x_grad_block_ptr, x_grad_tile, boundary_check=(0, 1))
        tl.store(weight_grad_partials_block_ptr, weight_grad_tile, boundary_check=(0, 1))

        # --- advancing ---
        x_block_ptr = tl.advance(x_block_ptr, (0, TILE_D))
        weight_block_ptr = tl.advance(weight_block_ptr, (TILE_D,))
        x_grad_block_ptr = tl.advance(x_grad_block_ptr, (0, TILE_D))
        weight_grad_partials_block_ptr = tl.advance(weight_grad_partials_block_ptr, (0, TILE_D))



class TritonWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """Compute weighted sum: (weight * x).sum(dim=-1).
        
        Args:
            x: (..., D) - input tensor
            weight: (D,) - weight vector
        Returns:
            output: (...,) - weighted sum along last dimension
        """
        # print("Triton weighted sum forward!!!")
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        N, D = x.shape[0], x.shape[1]
        TILE_N, TILE_D = 16, triton.next_power_of_2(D) // 16

        ctx.save_for_backward(x, weight)
        ctx.input_shape = input_shape
        ctx.TILE_N = TILE_N
        ctx.TILE_D = TILE_D

        output = torch.empty(N, device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(N, TILE_N),)
        weighted_sum_fwd[grid](
            # x: (N, D)
            x_ptr=x, 
            x_stride_0=x.stride(0), 
            x_stride_1=x.stride(1),
            # weight: (D,) 
            weight_ptr=weight,
            weight_stride=weight.stride(0),
            # output: (N,)
            output_ptr=output,
            output_stride=output.stride(0),
            # shapes
            N=N, D=D, 
            TILE_N=TILE_N, TILE_D=TILE_D
        )
        
        return output.view(input_shape[:-1])
    
    @staticmethod
    def backward(ctx, outgrad):
        """Compute gradients w.r.t. x and weight.
        
        Returns:
            x_grad: (..., D) - outgrad[:, None] * weight[None, :]
            weight_grad: (D,) -  x.T @ outgrad
        """

        # read data
        output_shape = outgrad.shape
        x, weight = ctx.saved_tensors # (N, D) and (D,)
        input_shape = ctx.input_shape # (..., D) with (...) = N
        N, D = x.shape[0], x.shape[1]
        TILE_N, TILE_D = ctx.TILE_N, ctx.TILE_D
        NUM_BLOCKS = triton.cdiv(N, TILE_N)
        outgrad = rearrange(outgrad, "... -> (...)") # (...,) -> (N,)
        # x_grad = outgrad[:, None] * weight[None, :]
        # weight_grad = x.T @ outgrad

        x_grad = torch.empty(*x.shape, device=x.device, dtype=x.dtype) # (N, D)
        weight_grad_partials = torch.empty(NUM_BLOCKS, D, 
            device=weight.device, dtype=weight.dtype)

        grid = (triton.cdiv(N, TILE_N),)
        weighted_sum_bwd[grid](
            # x: (N, D)
            x_ptr=x, 
            x_stride_0=x.stride(0), 
            x_stride_1=x.stride(1),
            # weight: (D,)
            weight_ptr=weight, 
            weight_stride=weight.stride(0),
            # outgrad: (N,)
            outgrad_ptr=outgrad, 
            outgrad_stride=outgrad.stride(0),
            # x_grad: (N, D)
            x_grad_ptr=x_grad,
            # weight_grad_partials: (NUM_BLOCKS, D)
            weight_grad_partials_ptr=weight_grad_partials, 
            weight_grad_partials_stride_0=weight_grad_partials.stride(0),
            weight_grad_partials_stride_1=weight_grad_partials.stride(1),
            # shapes
            N=N, D=D, NUM_BLOCKS=NUM_BLOCKS,
            TILE_N=TILE_N, TILE_D=TILE_D
        )
        
        outgrad = outgrad.view(output_shape)
        weight_grad = torch.sum(weight_grad_partials, axis=0)
        return x_grad.view(input_shape), weight_grad

class PytorchWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """Compute weighted sum: (weight * x).sum(dim=-1).
        
        Args:
            x: (..., D) - input tensor
            weight: (D,) - weight vector
        Returns:
            output: (...,) - weighted sum along last dimension
        """
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        output = (weight * x).sum(axis=-1)

        ctx.save_for_backward(x, weight)
        ctx.input_shape = input_shape
        
        return output.view(input_shape[:-1])
    
    @staticmethod
    def backward(ctx, outgrad):
        """Compute gradients w.r.t. x and weight.
        
        Returns:
            x_grad: (..., D) - outgrad[:, None] * weight[None, :]
            weight_grad: (D,) -  x.T @ outgrad
        """
        x, weight = ctx.saved_tensors
        input_shape = ctx.input_shape
        output_shape = outgrad.shape

        outgrad = rearrange(outgrad, "... -> (...)")
        x_grad = outgrad[:, None] * weight[None, :]
        weight_grad = x.T @ outgrad

        outgrad = outgrad.view(output_shape)
        return x_grad.view(input_shape), weight_grad


def ref_weighted_sum(x, weight):
    return (weight * x).sum(axis=-1)

SHAPES = [
    (50, 100), 
    (30, 40, 100), 
    (20, 30, 40, 100),
    (10, 20, 30, 40, 1000)
]

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_weighted_sum_forward(shape, dtype):
    """Test TritonWeightedSum forward pass."""
    device = torch.device("cuda:0")
    
    x = torch.rand(*shape, device=device, dtype=dtype)
    weight = torch.rand(shape[-1], device=device, dtype=dtype)
    
    out = TritonWeightedSum.apply(x, weight)
    out_ref = ref_weighted_sum(x, weight)
    
    torch.testing.assert_close(out, out_ref)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_weighted_sum_backward(shape, dtype):
    """Test TritonWeightedSum backward pass."""
    device = torch.device("cuda:0")
    
    x = torch.rand(*shape, device=device, dtype=dtype, requires_grad=True)
    weight = torch.rand(shape[-1], device=device, dtype=dtype, requires_grad=True)
    
    x_ref = x.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach().requires_grad_(True)
    
    out = TritonWeightedSum.apply(x, weight)
    out_ref = ref_weighted_sum(x_ref, weight_ref)
    
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)
    
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-3, atol=1e-3)