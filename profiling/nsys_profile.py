import cs336_basics as lib
import cs336_basics.model as nn
import cs336_basics.optimizer as optim
import torch
import torch.cuda.nvtx as nvtx
import timeit
import argparse
import logging
import json
from typing import Callable, Dict, Any
from pathlib import Path
from nn_utils import cross_entropy

# Model configuration dictionaries
CONFIGS = json.load(open("model_configs.json", "r"))
logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
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
        '--num-warmups', 
        type=int, 
        default=10,
        help='Number of warmup steps'
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
        '--mode', 
        choices=['forward', 'grad', 'train'], 
        default='forward',
        help=("Running modes:\n"
            "  `forward`: Running forward pass only.\n"
            "  `grad`: Running forward and backward passes.\n"
            "  `train`: Running full training steps."
        )
    )

    parser.add_argument(
        '--annotate_attention', 
        action='store_true',
        help='Add NVTX annotatton to scaled dot product attention'
    )

    return parser.parse_args()

def profiling(args):

    # Define a model (with random weights)
    cfg = CONFIGS[args.config]
    logger.info(f"Benchmarking config: {args.config}")
    logger.info(f"Model parameters: {cfg}")
    device = get_device(args.gpu_index)
    
    with nvtx.range("define_model"):
        if args.annotate_attention:
            logger.info("Replacing scaled_dot_product_attention to NVTX annotated version.")
            from attention import annotated_scaled_dot_product_attention
            nn.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        model = nn.BasicsTransformerLM(**cfg).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model {args.config} loaded with {num_params:,} parameters")
    
    # Initialize optimizer if requested
    if args.mode == "train":
        optimizer = optim.AdamW(model.parameters())

    # Define an input and output
    with nvtx.range("define_input_output"):
        vocab_size = model.vocab_size
        X = torch.randint(
            high=vocab_size, size=(args.batch_size, args.sequence_length),
            device=next(model.parameters()).device
        )
        labels = torch.randint(0, vocab_size, (args.batch_size, args.sequence_length))
        labels = labels.to(next(model.parameters()).device)

    def _forward():
        model.zero_grad(set_to_none=True)
        with nvtx.range("forward"):
            logits = model(X)
            loss = cross_entropy(logits, labels)

    def _grad():
        model.zero_grad(set_to_none=True)
        with nvtx.range("forward"):
            logits = model(X) # (args.batch_size, args.sequence_length, vocab_size)
            loss = cross_entropy(logits, labels)
        with nvtx.range("backward"):
            loss.backward()

    def _train():
        optimizer.zero_grad()
        with nvtx.range("forward"):
            logits = model(X) # (args.batch_size, args.sequence_length, vocab_size)
            loss = cross_entropy(logits, labels)
        with nvtx.range("backward"):
            loss.backward()
        with nvtx.range("optimizer_step"):
            optimizer.step()

    RUN_MAPPER = {
        "forward": _forward,
        "grad": _grad,
        "train": _train
    }
    run = RUN_MAPPER[args.mode]

    for _ in range(args.num_warmups): run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    for step in range(args.num_steps):
        # # start profiling after 10 warmup iterations
        # if step > args.num_warmups: 
        #     torch.cuda.cudart().cudaProfilerStart()
        nvtx.range_push(f"step_{step}")
        run()
        nvtx.range_pop()

def main():
    args = parse_args()
    setup_logging(log_level=args.log_level)
    profiling(args)

if __name__ == "__main__":
    main()
    