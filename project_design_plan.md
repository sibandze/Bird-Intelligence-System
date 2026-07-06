> **First make the model learnable, then make it intelligent (contrastive).**

If we try to start with contrastive learning too early, we'll end up debugging multiple moving parts simultaneously:

* Data loading
* Spectrogram generation
* Dataset logic
* Transformer architecture
* Training loop
* Contrastive loss

When training doesn't work, we won't know which component is responsible.

Instead, build the project in layers:

### Phase 1 — Data Pipeline

* Load metadata (`train.csv`)
* Build label mappings
* Generate and save Mel spectrograms (`.npy`)
* Create Dataset and DataLoader classes
* Verify shapes and labels

**Success criterion:** You can iterate through batches reliably.

---

### Phase 2 — Audio Transformer Input Pipeline

* Spectrogram patching
* Patch embeddings
* Positional encodings

**Success criterion:** Input tensors become transformer-ready token sequences.

---

### Phase 3 — Audio Transformer Encoder

* Multi-head self-attention
* Feed-forward blocks
* Layer normalization
* Residual connections
* Stack multiple encoder layers

**Success criterion:** Forward pass works and produces meaningful embeddings.

---

### Phase 4 — Classification Baseline

* Add CLS token
* Add classification head
* Train on species labels
* Track loss and validation metrics

**Success criterion:** Model learns and beats random guessing.

---

### Phase 5 — Evaluation and Inference

* Validation pipeline
* Prediction generation
* Confusion analysis
* Submission formatting

**Success criterion:** End-to-end system trains and evaluates correctly.

---

### Phase 6 — Contrastive Learning (Most Important Stage)

Only after the above is stable:

* Create positive and negative pairs
* Build contrastive dataset
* Implement InfoNCE loss
* Pretrain encoder
* Fine-tune classifier on learned embeddings
* Compare against baseline

**Success criterion:** Contrastive pretraining improves downstream classification performance.

---

At that point, the contrastive learning component becomes a focused research contribution rather than a debugging exercise. We can clearly demonstrate:

1. A working supervised baseline.
2. A custom audio transformer architecture.
3. A contrastive pretraining strategy for weakly labeled bird audio.
4. Quantitative improvements from representation learning.
