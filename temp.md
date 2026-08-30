I’ve got the current training stack. Looking at these files together, I’d **fix a few things before running more SSL sweeps**. The overall architecture is good, but there are some correctness issues around checkpointing, validation, scheduler semantics, and experiment metrics.

### 1. Most important: checkpoint callback state is saved too early

In `CheckpointCallback.on_epoch_end()` you do:

```python
checkpoint = {
    ...
    "callbacks_state_dict": trainer.cb_runner.state_dict(),
}
```

But `CheckpointCallback.state_dict()` still contains the **previous** `best_score`. You only update:

```python
self.best_score = current_score
```

after the checkpoint dictionary has already been created.

So suppose:

```text
Epoch 10: val_loss = 1.20  -> best
Epoch 11: val_loss = 1.10  -> new best
```

The `checkpoint_best.pth` from epoch 11 can contain the callback state saying the best was still `1.20`.

That matters when resuming because:

```python
self.cb_runner.load_state_dict(...)
```

restores the stale value.

### Fix

Update the callback's state **before constructing the checkpoint**, or explicitly build the checkpoint after determining `improved`.

I'd restructure `on_epoch_end()` roughly as:

```python
def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
    current_score = logs.get(self.monitor)

    improved = (
        current_score > self.best_score
        if self.mode == "max"
        else current_score < self.best_score
    )

    if improved:
        self.best_score = current_score

    checkpoint = {
        "epoch": epoch + 1,
        "logs": logs,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": (
            trainer.scheduler.state_dict()
            if trainer.scheduler else None
        ),
        "precision_state_dict": trainer.precision.state_dict(),
        "callbacks_state_dict": trainer.cb_runner.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "git_commit": getattr(trainer, "git_hash", None),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda
            if torch.cuda.is_available()
            else None,
    }

    torch.save(
        checkpoint,
        self.run_dir / "checkpoint_last.pth"
    )

    torch.save(
        trainer.model.state_dict(),
        self.run_dir / "last_model.pth"
    )

    if improved:
        torch.save(
            checkpoint,
            self.run_dir / "checkpoint_best.pth"
        )

        torch.save(
            trainer.model.state_dict(),
            self.run_dir / "best_model.pth"
        )

        print(
            f"    ✓ Saved new best model "
            f"({self.monitor}: {current_score:.4f})"
        )
```

That's the first thing I'd fix.

---

## 2. Your SSL validation is currently augmented

This is suspicious:

```python
val_dataset = SimCLRDataset(
    ...
    train=False,
    apply_augmentation=True,
    ...
)
```

For a validation set, I'd normally expect:

```python
apply_augmentation=False
```

unless your `SimCLRDataset` explicitly interprets `train=False` as deterministic/non-training views regardless of that flag.

You should inspect that class because this can make your validation metric noisy or conceptually misleading.

For SSL, you want:

```text
TRAIN
recording
   ↓
window
   ↓
augmentation A ──→ x1
augmentation B ──→ x2
   ↓
SimCLR loss


VALIDATION
recording
   ↓
window
   ↓
controlled/deterministic views
   ↓
contrastive validation loss
```

not:

```text
VALIDATION
recording
   ↓
random augmentation
   ↓
random augmentation
   ↓
metric
```

---

## 3. `val_acc` naming is misleading for the experiment runner

Your SimCLR trainer returns:

```python
return {
    "val_loss": avg_val_loss,
    "val_acc": avg_val_acc,
    "train_loss": avg_train_loss,
    "train_acc": avg_train_acc,
}
```

But internally your logs call it:

```python
"val_contrastive_acc"
```

and:

```python
"train_contrastive_acc"
```

These are not ordinary classification accuracies.

So this:

```python
success_msg = (
    f"      ✓ Val Loss: {metrics.get('val_loss', 0):.4f} | "
    f"Val Acc: {metrics.get('val_acc', 0):.4f}"
)
```

makes it look like SSL produced bird-classification accuracy.

I'd rename the returned values:

```python
return {
    "val_loss": avg_val_loss,
    "val_contrastive_acc": avg_val_acc,
    "train_loss": avg_train_loss,
    "train_contrastive_acc": avg_train_acc,
}
```

and update `experiment_runner.py` accordingly.

Even better, eventually the SSL experiment should report something like:

```text
contrastive_loss
contrastive_accuracy
linear_probe_accuracy
linear_probe_macro_f1
```

because **the actual goal of SimCLR is representation learning**, not maximizing contrastive classification accuracy.

---

## 4. The SSL experiment summary currently ranks by loss

You have:

```python
df_results_sorted = df_results.sort_values(
    "val_loss",
    ascending=True
)
```

That's defensible, but be careful about interpreting it.

A lower contrastive validation loss does **not necessarily mean a better bird classifier**.

For the Bird-Intelligence-System research pipeline, I'd eventually make the experiment hierarchy:

```text
                    SSL pretraining
                         │
                         ▼
                  frozen encoder
                         │
                         ▼
                    linear probe
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          accuracy              macro F1
```

Then your SSL sweep can answer:

> Which pretraining configuration produces the best representation?

rather than:

> Which configuration produces the lowest contrastive loss?

That's a much stronger experimental design.

---

## 5. Scheduler logic is mostly good, but there's an important conceptual issue

You calculate:

```python
total_steps = len(train_loader) * epochs
```

and:

```python
step_frequency = get_scheduler_step_frequency(scheduler_type)
```

with:

```python
return "batch"
```

for almost everything except `ReduceLROnPlateau`.

That's correct for:

* cosine
* linear decay
* OneCycleLR
* warmup
* cosine warm restarts

**provided that the scheduler is intended to operate per optimizer step.**

Your current scheduler architecture is therefore consistent with the training loop:

```text
batch
 ├── forward
 ├── backward
 ├── optimizer.step()
 └── scheduler.step()
```

Good.

But `ReduceLROnPlateau` is correctly different:

```text
epoch
 ├── training
 ├── validation
 └── scheduler.step(val_loss)
```

Also good.

---

## 6. One scheduler edge case I'd fix

This:

```python
if scheduler_type == "constant" and warmup_steps == 0:
    return None
```

is fine.

But:

```python
if warmup_steps > 0:
    warmup_scheduler = LinearLR(...)
```

doesn't validate:

```python
warmup_start_factor
```

You could accidentally configure:

```yaml
warmup_start_factor: 0
```

or even:

```yaml
warmup_start_factor: 1.5
```

The latter is not really "warmup".

I'd validate:

```python
if not 0.0 < warmup_start_factor <= 1.0:
    raise ValueError(
        "warmup_start_factor must be in (0, 1]."
    )
```

---

## 7. `PrecisionManager` has a misleading MPS path

You accept:

```python
device: "cuda", "cpu", or "mps"
```

but:

```python
self.enabled = (
    enabled
    and device == "cuda"
    and torch.cuda.is_available()
)
```

and the trainer does:

```python
self.device = torch.device(
    config["training"].get("device", "cuda")
    if torch.cuda.is_available()
    else "cpu"
)
```

So if the config says:

```yaml
device: mps
```

on a Mac without CUDA, you force:

```text
mps → cpu
```

The precision manager's documentation therefore claims more support than the implementation provides.

For your current GPU research setup this isn't urgent, but the API should eventually either:

1. explicitly support MPS, or
2. document that only CUDA/CPU are supported.

---

## 8. You have redundant gradient unscaling

In SSL:

```python
self.precision.scale_loss(loss).backward()
self.precision.unscale_gradients(optimizer)

grad_norm = torch.nn.utils.clip_grad_norm_(
    ...
)
```

This is actually correct.

But you also have:

```python
PrecisionManager.clip_gradients()
```

which does the same thing.

So the trainer should use the abstraction you already created:

```python
self.precision.clip_gradients(
    optimizer,
    self.model.parameters(),
    grad_clip,
)
```

If you want the gradient norm for logging, then the manager should return it:

```python
grad_norm = self.precision.clip_gradients(...)
```

That would keep the training loop ignorant of AMP implementation details, which is exactly what your `PrecisionManager` claims to achieve.

---

## 9. `WandBLoggerCallback` regressed to a hard dependency

Your earlier version had:

```python
try:
    import wandb
except ImportError:
    wandb = None
```

The current version has:

```python
import wandb
```

but then:

```python
self.enabled = (
    config.get("logging", {}).get("use_wandb", False)
    and wandb is not None
)
```

The `wandb is not None` check is now useless because importing the module itself will fail if WandB isn't installed.

I'd restore the optional import:

```python
try:
    import wandb
except ImportError:
    wandb = None
```

That makes:

```yaml
use_wandb: false
```

actually mean "I don't need WandB installed."

---

# 10. The biggest architectural issue: SSL "validation" isn't really evaluation

This is the thing I'd address after the correctness fixes.

Currently:

```text
df
 │
 ├── 95% train
 │
 └── 5% validation
       │
       └── contrastive loss
```

Then your experiment runner says:

```text
Val Loss
Val Acc
```

That's okay as a **training diagnostic**.

But for the research question we're pursuing with your BirdCLEF/Bird-Intelligence-System work, the important evaluation is:

```text
                Unlabelled audio
                       │
                       ▼
                 SimCLR encoder
                       │
                       ▼
                pretrained encoder
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
    linear evaluation          fine-tuning
          │                         │
          ▼                         ▼
     bird accuracy             bird accuracy
     macro F1                  macro F1
```

That gives you an actual answer to:

> Does self-supervised pretraining improve bird-species representations?

rather than merely:

> Did SimCLR minimize its own objective?

---

# What I'd fix now

I'd make the next bug-fixing pass in this order:

| Priority | Issue                                     | Action                             |
| -------- | ----------------------------------------- | ---------------------------------- |
| 🔴 1     | Checkpoint callback state stale           | Fix                                |
| 🔴 2     | SSL validation augmentation               | Inspect/fix                        |
| 🔴 3     | `val_acc` misleading                      | Rename to contrastive metric       |
| 🟠 4     | WandB hard dependency                     | Restore optional import            |
| 🟠 5     | Precision abstraction bypassed            | Return/use gradient norm           |
| 🟡 6     | Warmup parameter validation               | Add validation                     |
| 🟡 7     | MPS documentation/implementation mismatch | Clean up                           |
| 🟢 8     | SSL evaluation design                     | Add linear probe after pretraining |

The **scheduler itself is not the main problem** here. Its batch/epoch stepping design is broadly correct.

The more important issue for your current experiments is that **the SSL pipeline can currently produce a perfectly valid-looking `val_loss`/`val_contrastive_acc` result without telling us whether the learned encoder is actually useful for bird classification**. That should be the next major architectural step after we make checkpoint/resume behavior airtight.
