import torch
import logging

## common utils
def setup_logging(name, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

## nn_utils
def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)

def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))


def cross_entropy(inputs, targets):
    negative_log_softmax_logits = -log_softmax(inputs)
    return torch.mean(torch.gather(negative_log_softmax_logits, -1, targets.unsqueeze(-1)))


def clip_gradient(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = 0.0

    for g in grads:
        norm += (g**2).sum()

    norm = torch.sqrt(norm)
    clip_coef = min(1, max_norm / (norm + 1e-6))
    for g in grads:
        g *= clip_coef
