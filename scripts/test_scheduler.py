"""Quick test to verify scheduler integration."""

import torch
import torch.optim as optim
from src.training.scheduler import create_scheduler

def test_schedulers():
    """Test all scheduler types."""
    model = torch.nn.Linear(10, 2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    scheduler_types = [
        "constant", "cosine", "linear_decay", 
        "cosine_warm_restarts", "reduce_on_plateau", "one_cycle"
    ]
    
    for sched_type in scheduler_types:
        print(f"\n{'='*50}")
        print(f"Testing: {sched_type} (with warmup=100)")
        
        # Reset optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
        
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_type=sched_type,
            warmup_steps=100,
            total_steps=1000,
        )
        
        if scheduler is None:
            print(f"  No scheduler created (constant without warmup)")
            continue
        
        # Simulate training steps
        lrs = []
        for step in range(200):
            if scheduler is not None:
                scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        
        print(f"  Initial LR: {lrs[0]:.6f}")
        print(f"  Final LR:   {lrs[-1]:.6f}")
        print(f"  Max LR:     {max(lrs):.6f}")
        print(f"  Min LR:     {min(lrs):.6f}")

if __name__ == "__main__":
    test_schedulers()
