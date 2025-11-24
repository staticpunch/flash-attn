import triton
import triton.language as tl
import torch
from einops import rearrange, einsum
import einx

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr, output_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)

    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    #   i.e., how many elements do I skip to move one step in each dimension?
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to load/store at a time
    # - The order of the dimensions in memory from major to minor
    #   (e.g., np.argsort(strides)) for optimizations, especially useful on H100

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )

    # Initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the current block pointer
        # Since ROWS_TILE_SIZE might not divide ROWS, and D_TILE_SIZE might not divide D,
        # we need boundary checks for both dimensions
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)

        # Compute the weighted sum of the row.
        # Output == tl.sum(row * weight[None, :], axis=1)
        output += tl.sum(row * weight[None, :], axis=1)

        # Move the pointers to the next tile
        # These are (row, columns) coordinate deltas
        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE)) # Move by D_TILE_SIZE in the last dimension
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,)) # Move by D_TILE_SIZE

    # Write output to the output block pointer (a single scalar per row).
    # Since ROWS_TILE_SIZE might not divide ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward(
    # Pointers to tensors
    grad_output_ptr, x_ptr, weight_ptr,
    grad_x_ptr, partial_grad_weight_ptr,
    # Stride information
    stride_go_r,
    stride_x_r, stride_x_d,
    stride_w_d,
    stride_gx_r, stride_gx_d,
    stride_pgw_r, stride_pgw_d,
    # Tensor dimensions
    NUM_ROWS, D,
    NUM_GRID_ROWS, # number of rows in the partial_grad_weight tensor
    # Kernel constants
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)

    # --- Corrected Block Pointers ---

    # grad_output is a 1D vector of shape (NUM_ROWS,)
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_go_r,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    # Load the gradient for the entire row-tile once, as it's constant across the D-dimension loop
    grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero") # shape: (ROWS_TILE_SIZE,)

    # x is a 2D matrix of shape (NUM_ROWS, D)
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_x_r, stride_x_d),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # weight is a 1D vector of shape (D,)
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_w_d,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # grad_x is a 2D matrix of shape (NUM_ROWS, D)
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gx_r, stride_gx_d),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    
    # partial_grad_weight is a 2D matrix of shape (NUM_GRID_ROWS, D)
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(NUM_GRID_ROWS, D),
        strides=(stride_pgw_r, stride_pgw_d),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # --- grad_x computation ---
        # grad_x = grad_output[:, None] * weight[None, :] (outer product)
        weight_tile = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # shape: (D_TILE_SIZE,)
        # Broadcasting grad_output: (ROWS_TILE_SIZE,) -> (ROWS_TILE_SIZE, 1)
        # Broadcasting weight_tile: (D_TILE_SIZE,) -> (1, D_TILE_SIZE,)
        grad_x_tile = grad_output[:, None] * weight_tile[None, :] # shape: (ROWS_TILE_SIZE, D_TILE_SIZE)
        tl.store(grad_x_block_ptr, grad_x_tile, boundary_check=(0, 1))

        # --- partial_grad_weight computation ---
        # grad_weight = x.T @ grad_output
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # shape: (ROWS_TILE_SIZE, D_TILE_SIZE)
        # Broadcasting grad_output: (ROWS_TILE_SIZE,) -> (ROWS_TILE_SIZE, 1)
        grad_weight_tile = tl.sum(x_tile * grad_output[:, None], axis=0, keep_dims=True) # shape: (1, D_TILE_SIZE)
        tl.store(partial_grad_weight_block_ptr, grad_weight_tile, boundary_check=(0, 1))

        # --- Advance pointers for the next iteration ---
        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))
        grad_x_block_ptr = tl.advance(grad_x_block_ptr, (0, D_TILE_SIZE))
        partial_grad_weight_block_ptr = tl.advance(partial_grad_weight_block_ptr, (0, D_TILE_SIZE))

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        input_shape = x.shape
        D = x.shape[-1]
        
        # Reshape input tensor to 2D
        x_2d = rearrange(x, "... d -> (...) d")
        n_rows = x_2d.shape[0]

        ctx.save_for_backward(x_2d, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        # Contiguous check on reshaped tensor
        assert x_2d.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        # These values need to be saved for the backward pass
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16 # Roughly 16 loops through the embedding dimension
        ctx.ROWS_TILE_SIZE = 16 # Each thread processes 16 batch elements at a time
        ctx.input_shape = input_shape
        
        # Need to initialise empty result tensor.
        y = torch.empty(n_rows, device=x.device, dtype=x.dtype)

        # Launch our kernel with n instances in our 1D grid.
        grid = (triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)
        weighted_sum_fwd[grid](
            x_2d, weight, y,
            x_2d.stride(0), x_2d.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        input_shape = ctx.input_shape
        n_rows, D = x.shape
        
        # grad_out comes in with the shape of the forward's output. Flatten it.
        grad_out = grad_out.contiguous().view(n_rows)

        # Our strategy is for each thread block to first write to a partial buffer,
        # then we reduce over this buffer to get the final gradient.
        n_grid_rows = triton.cdiv(n_rows, ROWS_TILE_SIZE)
        partial_grad_weight = torch.empty((n_grid_rows, D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        grid = (n_grid_rows,)
        
        # --- Corrected kernel call ---
        weighted_sum_backward[grid](
            # Tensors
            grad_out, x, weight, grad_x, partial_grad_weight,
            # Strides
            grad_out.stride(0),
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            # Dimensions
            NUM_ROWS=n_rows, D=D,
            NUM_GRID_ROWS=n_grid_rows,
            # Kernel constants
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )

        grad_weight = partial_grad_weight.sum(axis=0)
        # Reshape grad_x back to original input shape
        return grad_x.view(input_shape), grad_weight

# Helper for calling the function
# weighted_sum = WeightedSumFunc.apply
def weighted_sum(x, weight):
    return WeightedSumFunc.apply(x, weight)


def ref_weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1)
    
### Test for Both Forward and Backward Pass
def test_weighted_sum_autograd():
    print("Testing Autograd Functionality...")
    # Using non-power-of-2 and multi-dimensional shapes to test reshaping and boundary checks
    B, T, D = 13, 27, 1567
    device = "cuda"

    # Create input tensors, requires_grad=True to track gradients
    x = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)
    
    # Create clones for PyTorch's native implementation
    x_torch = x.clone().detach().requires_grad_(True)
    weight_torch = weight.clone().detach().requires_grad_(True)

    # --- Forward pass ---
    output_triton = weighted_sum(x, weight)
    output_torch = ref_weighted_sum(x_torch, weight_torch)

    torch.testing.assert_close(output_triton, output_torch, rtol=1e-5, atol=1e-5)
    print("✅ Forward Pass Passed!")

    # --- Backward pass ---
    # Use a random gradient for the output to test the chain rule
    grad_output = torch.randn_like(output_triton)
    output_triton.backward(grad_output)
    output_torch.backward(grad_output)

    # Compare gradients
    torch.testing.assert_close(x.grad, x_torch.grad, rtol=1e-5, atol=1e-5)
    print("✅ Backward Pass (grad_x) Passed!")
    torch.testing.assert_close(weight.grad, weight_torch.grad, rtol=1e-5, atol=1e-5)
    print("✅ Backward Pass (grad_weight) Passed!")


if __name__ == "__main__":
    test_weighted_sum_autograd()