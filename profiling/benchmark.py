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
from functools import wraps
from contextlib import nullcontext
from nn_utils import cross_entropy

# Model configuration dictionaries
CONFIGS = json.load(open("model_configs.json", "r"))
logger = logging.getLogger("benchmarking")

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

def mean(x: list[float]) -> float:
    return sum(x) / len(x)

def std(x: list[float]) -> float:
    if len(x) <= 1:
        return 0.0
    mu = mean(x)
    return (sum((xi - mu) ** 2 for xi in x) / (len(x) - 1)) ** 0.5


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str,
        choices=list(CONFIGS.keys()), default="small")
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--log-level', type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-warmups', type=int, default=10)
    parser.add_argument('--num-steps', type=int, default=20)
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--dtype', type=str, default='fp32', 
        choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--compile', action='store_true',
        help="Compile model using torch.compile()")
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

    return parser.parse_args()

def benchmarking(args):

    # Define a model (with random weights)
    cfg = CONFIGS[args.config]
    logger.info(f"Benchmarking config: {args.config}")
    logger.info(f"Model parameters: {cfg}")
    device = get_device(args.gpu_index)
    

    model = nn.BasicsTransformerLM(**cfg).to(device)
    if args.compile:
        logger.info("Compiling model with `torch.compile()`")
        model = torch.compile(model)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model {args.config} loaded with {num_params:,} parameters")
    
    # Initialize optimizer if requested
    if args.mode == "train":
        optimizer = optim.AdamW(model.parameters())

    # Define an input and output
    vocab_size = model.vocab_size
    X = torch.randint(
        high=vocab_size, size=(args.batch_size, args.sequence_length),
        device=next(model.parameters()).device
    )
    labels = torch.randint(0, vocab_size, (args.batch_size, args.sequence_length))
    labels = labels.to(next(model.parameters()).device)

    def _forward():
        model.zero_grad(set_to_none=True)
        logits = model(X) # (args.batch_size, args.sequence_length, vocab_size)
        loss = cross_entropy(logits, labels)

    def _grad():
        model.zero_grad(set_to_none=True)
        logits = model(X) # (args.batch_size, args.sequence_length, vocab_size)
        loss = cross_entropy(logits, labels)
        loss.backward()

    def _train():
        optimizer.zero_grad()
        logits = model(X) # (args.batch_size, args.sequence_length, vocab_size)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

    RUN_MAPPER = {
        "forward": _forward,
        "grad": _grad,
        "train": _train
    }
    run = RUN_MAPPER[args.mode]
    
    with make_context(args.dtype):
        logger.info(f"Running {args.num_warmups} warmup iterations...")
        for _ in range(args.num_warmups): run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
        logger.info(f"Running {args.num_steps} benchmark iterations...")
        times: list[float] = []
        for step in range(args.num_steps):
            start_time = timeit.default_timer()
            run()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = timeit.default_timer()
            times.append((end_time - start_time) * 1000)
        
    statistics = dict(
        mean=mean(times),
        std=std(times),
        min=min(times),
        max=max(times)
    )
    statistics = {k: f"{v:6.2f}" for k, v in statistics.items()}
    return statistics

def memory_profiling(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping memory profiling")
            return func(*args, **kwargs)
            
        logger.info("Enabled memory profiling.")
        torch.cuda.memory._record_memory_history(max_entries=100000) # Start recording memory history
        result = func(*args, **kwargs)
        ## Output handling
        output_dir = Path("memory_snapshots")
        output_dir.mkdir(exist_ok=True)
        snapshot_file = output_dir / \
            f"model_{args[0].config}-dtype_{args[0].dtype}-mode_{args[0].mode}.pickle"
        torch.cuda.memory._dump_snapshot(snapshot_file)
        torch.cuda.memory._record_memory_history(enabled=None) # Stop recording history.
        logger.info(f"Saved memory snapshot to {snapshot_file}")
        return result
    return wrapper

def make_context(dtype):
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }   
    dtype = dtype_map[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    context = nullcontext()
    if dtype == torch.float32:
        logger.info("Running with full precision (fp32).")
    else:
        context = torch.autocast(device_type=device, dtype=dtype)
        logger.info(f"Running with mixed precision ({dtype}).")
    return context

def main():
    args = parse_args()
    setup_logging(log_level=args.log_level)

    run = memory_profiling(benchmarking) if args.memory else benchmarking        
    statistics = run(args)
    logger.info(f"Running time:\n  {statistics}")

if __name__ == "__main__":
    main()
    