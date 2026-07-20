"""Learning rate schedulers for transformer training."""

import torch.optim as optim
from typing import Optional


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 0,
    total_steps: int = 1000,
    min_lr: float = 1e-6,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Creates a learning rate scheduler with optional warmup.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_type: One of ['constant', 'cosine', 'linear_decay', 'cosine_warm_restarts', 'reduce_on_plateau', 'one_cycle']
        warmup_steps: Number of warmup steps (uses linear warmup)
        total_steps: Total training steps for schedulers that need it
        min_lr: Minimum learning rate for decay schedulers
    
    Returns:
        A PyTorch scheduler, or None if 'constant' with no warmup
    """
    
    # If no scheduler needed
    if scheduler_type == "constant" and warmup_steps == 0:
        return None
    
    # Create warmup scheduler if specified
    if warmup_steps > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1,  # Start at 10% of lr
            end_factor=1.0,     # End at full lr
            total_iters=warmup_steps
        )
    
    # Create main scheduler
    if scheduler_type == "cosine":
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr
        )
    elif scheduler_type == "linear_decay":
        main_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr,
            total_iters=total_steps - warmup_steps
        )
    elif scheduler_type == "cosine_warm_restarts":
        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=total_steps // 3,  # Restart every 1/3 of training
            T_mult=2,
            eta_min=min_lr
        )
    elif scheduler_type == "reduce_on_plateau":
        # This requires validation loss, handled separately in training loop
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=min_lr,
            verbose=True
        )
    elif scheduler_type == "one_cycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps if total_steps > 0 else 0.3,
            anneal_strategy='cos',
            final_div_factor=1e4
        )
    elif scheduler_type == "constant":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Combine warmup and main scheduler
    if warmup_steps > 0:
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
    
    return main_scheduler


def get_scheduler_step_frequency(scheduler_type: str) -> str:
    """
    Returns whether scheduler steps per 'epoch' or 'batch'.
    
    Args:
        scheduler_type: The scheduler type string
    
    Returns:
        'epoch' for ReduceLROnPlateau, 'batch' for everything else
    """
    if scheduler_type == "reduce_on_plateau":
        return "epoch"
    return "batch"
