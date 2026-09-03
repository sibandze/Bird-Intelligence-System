We need to ensure that each experiment uses a controlled, fixed-size dataset – exactly num_classes classes and num_samples_per_class samples per class – regardless of the full metadata file size. This guarantees fair comparisons across sweeps and eliminates variability introduced by class imbalance or differing sample counts.

Currently, ExperimentManager.load_data() just loads the whole CSV. We should modify it to filter and downsample the dataset once, cache the result, and pass that subset to all training runs.

---

Recommended Changes

1. Update load_data() in ExperimentManager

Add logic to:

· Read the full CSV.
· Identify the label column (assume 'label' by default; make configurable).
· Keep only classes that have at least num_samples_per_class examples.
· Select the top num_classes most frequent classes (or random, but sorted gives reproducibility).
· For each selected class, randomly sample num_samples_per_class rows (using the set seed).
· Store the filtered DataFrame as self.df.

```python
def load_data(self):
    """Load and distill dataset to num_classes and num_samples_per_class."""
    if self.df is None:
        csv_path = resolve_metadata_csv_path(self.base_config)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Processed CSV not found at {csv_path}. Run data pipeline first: python main.py --pipeline"
            )

        full_df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(full_df)} samples from {csv_path}")

        # ---- Distillation ----
        data_cfg = self.base_config.get('data', {})
        num_classes = data_cfg.get('num_classes')
        num_samples_per_class = data_cfg.get('num_samples_per_class')
        label_col = data_cfg.get('label_column', 'label')   # allow config override

        if num_classes and num_samples_per_class:
            print(f"🔬 Distilling dataset to {num_classes} classes x {num_samples_per_class} samples each")

            # Group by label
            grouped = full_df.groupby(label_col)
            class_counts = grouped.size().sort_values(ascending=False)

            # Select top num_classes with enough samples
            eligible_classes = class_counts[class_counts >= num_samples_per_class]
            if len(eligible_classes) < num_classes:
                raise ValueError(
                    f"Only {len(eligible_classes)} classes have at least {num_samples_per_class} samples. "
                    f"Need {num_classes}. Reduce num_classes or increase data."
                )

            selected_classes = eligible_classes.index[:num_classes].tolist()
            print(f"   Selected classes: {selected_classes}")

            # Sample per class
            sampled_dfs = []
            rng = np.random.RandomState(self.seed)   # ensure reproducibility
            for cls in selected_classes:
                cls_df = full_df[full_df[label_col] == cls]
                sampled = cls_df.sample(n=num_samples_per_class, random_state=rng)
                sampled_dfs.append(sampled)

            self.df = pd.concat(sampled_dfs, ignore_index=True)
            print(f"✓ Final distilled dataset: {len(self.df)} samples")

            # Optional: shuffle the final dataset
            self.df = self.df.sample(frac=1, random_state=rng).reset_index(drop=True)
        else:
            # If no distillation requested, use full dataset
            self.df = full_df

    return self.df
```

2. Add seed attribute to ExperimentManager

The filtering uses self.seed (set via set_seed()). Add self.seed = seed in set_seed() and initialize it in __init__.

3. Make label column configurable (optional)

Add to config.yaml under data:

```yaml
data:
  label_column: "species"   # or "label", depending on your CSV
```

---

Why This Approach?

· Reproducibility – sampling uses a fixed random seed, so every run uses the same subset.
· Controlled comparisons – hyperparameter sweeps are not confounded by changing class sets or sample counts.
· Efficiency – the distilled DataFrame is built once and reused for all runs in the experiment.
· Flexibility – you can still keep the full CSV for other purposes (e.g., pretraining, full‑scale evaluations).

---

Edge Cases to Handle

· Not enough classes – raise a clear error suggesting to reduce num_classes or collect more data.
· Classes with too few samples – they are excluded; only the most frequent, adequately sampled classes are used.
· Different label column names – provide a config entry to avoid hardcoding.

---

Final Note

After this change, every training run in the sweep will use exactly the same balanced subset, making accuracy/F1 scores directly comparable. The full metadata file remains untouched, ready for other uses.
