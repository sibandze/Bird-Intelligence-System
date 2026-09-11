#!/usr/bin/env bash
# scripts/colab_setup.sh
set -euo pipefail

# ── Config ──────────────────────────────────────────────
SESSION="${1:-bis}"
GPU="${2:-T4}"
REPO_URL="https://github.com/sibandze/Bird-Intelligence-System.git"
BRANCH="dev-unsupervised"
REPO_DIR="/content/Bird-Intelligence-System"

echo "=== Step 1: Provision VM ==="
colab new -s "$SESSION" --gpu "$GPU" --high-mem

echo "=== Step 2: Mount Google Drive ==="
colab drivemount -s "$SESSION"

echo "=== Step 3: Clone repo + install deps ==="
colab exec -s "$SESSION" <<PYEOF
import subprocess, os, sys

REPO_DIR = "$REPO_DIR"
BRANCH = "$BRANCH"
REPO_URL = "$REPO_URL"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=REPO_DIR, check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-r",
    os.path.join(REPO_DIR, "requirements.txt")], check=True)
print("Repo ready and deps installed.")
PYEOF

echo "=== Step 4: Restore data from Drive ==="
colab upload -s "$SESSION" scripts/colab_restore_data.py "$REPO_DIR/scripts/colab_restore_data.py"
colab exec -s "$SESSION" -f scripts/colab_restore_data.py

echo "=== Step 5: Verify ==="
colab exec -s "$SESSION" <<PYEOF
import torch, os

REPO = "$REPO_DIR"
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

drive = "/content/drive/MyDrive"
print(f"Drive: {os.listdir(drive)[:5]}")

for label, rel in [("Audio", "data/raw_audio"),
                    ("Spectrograms", "data/processed_spectrograms"),
                    ("Metadata", "data/metadata")]:
    full = os.path.join(REPO, rel)
    if os.path.isdir(full):
        n = len([f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))])
        print(f"  {label}: {n} file(s)")
    else:
        print(f"  {label}: missing")
PYEOF

echo ""
echo "=== Setup complete ==="
echo "Session: $SESSION | GPU: $GPU"
echo ""
echo "Next steps:"
echo "  bash scripts/colab_experiment.sh ssl_sanity --dry-run"
echo "  bash scripts/colab_experiment.sh ssl_standard"
echo "  colab console -s $SESSION    # interactive shell"
echo "  colab stop -s $SESSION       # teardown when done"
