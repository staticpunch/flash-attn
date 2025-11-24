import torch
from torch.profiler import ProfilerActivity, record_function
import cs336_basics as lib
import cs336_basics.model as nn
import cs336_basics.optimizer as optim
import torch
import timeit
import argparse
import logging
import json
from typing import Callable, Dict, Any
from pathlib import Path
from nn_utils import cross_entropy

# Model configuration dictionaries
CONFIGS = json.load(open("model_configs.json", "r"))

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{index}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(index)}")
        return device
    else:
        logger.info("CUDA not available, using CPU")
        return torch.device("cpu")


def annotated_block_forward(self, x: torch.Tensor):
    """
    Args:
        x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
            The input to process with the Transformer block.

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """
    # NOTE: this is a pre-norm Transformer, and differs from the original
    # description in the paper.
    # Apply the multi-head self-attention sublayer
    with torch.profiler.record_function("BLOCK_LN1"):
        x_norm1 = self.ln1(x)
    
    with torch.profiler.record_function("BLOCK_ATTN"):
        x_attn = self.attn(x_norm1)
    
    with torch.profiler.record_function("BLOCK_ATTN_RESIDUAL"):
        attn_sublayer_output = x + x_attn

    # Apply the feed-forward sublayer
    with torch.profiler.record_function("BLOCK_LN2"):
        x_norm2 = self.ln2(attn_sublayer_output)
    
    with torch.profiler.record_function("BLOCK_FFN"):
        x_ffn = self.ffn(x_norm2)
    
    with torch.profiler.record_function("BLOCK_FFN_RESIDUAL"):
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        
    return ffn_sublayer_output

def run_training(args):

    # Define a model (with random weights)
    cfg = CONFIGS[args.config]
    logger.info(f"Benchmarking config: {args.config}")
    logger.info(f"Model parameters: {cfg}")
    device = get_device(args.gpu_index)

    if True:
        logger.info("Replacing transformer block forward with annotated version")
        nn.TransformerBlock.forward = annotated_block_forward
        
    model = nn.BasicsTransformerLM(**cfg).to(device)
    
    # Initialize optimizer if requested
    if args.use_optimizer:
        optimizer = optim.AdamW(model.parameters())

    # Define an input and output
    vocab_size = model.vocab_size
    X = torch.randint(
        high=vocab_size, size=(args.batch_size, args.sequence_length),
        device=next(model.parameters()).device
    )
    labels = torch.randint(0, vocab_size, (args.batch_size, args.sequence_length))
    labels = labels.to(next(model.parameters()).device)

    def annotated_training_step():
        # Zero gradients
        if args.use_optimizer:
            optimizer.zero_grad()
        else:
            model.zero_grad(set_to_none=True)
    
        # Forward
        with record_function("FORWARD"):
            logits = model(X) # (args.batch_size, args.sequence_length, vocab_size)
            loss = cross_entropy(logits, labels)
    
        # Backward
        with record_function("BACKWARD"):
            loss.backward()
    
        # Optimizer step if enabled
        if args.use_optimizer:
            with record_function("OPTIMIZER"):
                #print(f"Step {step}, loss: {y.item():.6f}")
                optimizer.step()
                
    def training_step():
        # Zero gradients
        if args.use_optimizer: optimizer.zero_grad()
        else: model.zero_grad(set_to_none=True)
    
        # Forward
        logits = model(X) # (args.batch_size, args.sequence_length, vocab_size)
        loss = cross_entropy(logits, labels)
    
        # Backward
        loss.backward()
    
        # Optimizer step if enabled
        if args.use_optimizer: optimizer.step() 

    # Run the model `num_steps` times
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=3, repeat=2),
            # record_shapes=True,
            profile_memory=True,
            with_stack=True, # Output stack trace for visualization
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True) # Needed to export stack trace for visualization
    ) as prof:
        for step in range(args.num_steps):
            training_step()
            prof.step()
    # Print out table
    table = prof.key_averages(
        # group_by_input_shape=True
    ).table(
        sort_by="cpu_time_total",
        max_name_column_width=80,
        row_limit=50
    )
    logger.info(table)
    with open("pytorch_profile.md", "w") as f:
        f.write(table)
    
    if True:
        text_path = f"torch_results/stacks.txt"
        svg_path = f"torch_results/stacks.svg"
        chrome_path = f"torch_results/trace.json"
        prof.export_stacks(text_path, "self_cuda_time_total")
        prof.export_chrome_trace(chrome_path)
        
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        choices=list(CONFIGS.keys()),
        default="small",
        help='Model configurations to benchmark'
    )
    
    parser.add_argument(
        '--gpu-index', 
        type=int, 
        default=0,
        help='GPU index to use (if available)'
    )

    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO',
        help='Logging level'
    )

    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1,
        help='Batch size for benchmarking'
    )
    
    parser.add_argument(
        '--num-steps', 
        type=int, 
        default=20,
        help='Number of steps in training loop'
    )
    
    parser.add_argument(
        '--sequence-length', 
        type=int, 
        default=128,
        help='Sequence length for benchmarking'
    )

    parser.add_argument(
        '--use-optimizer', 
        action='store_true',
        help='Include optimizer.step()'
    )

    return parser.parse_args()

def main():
    args = parse_args()
    global logger
    logger = setup_logging(log_level=args.log_level)
    run_training(args)

if __name__ == "__main__":
    main()
    