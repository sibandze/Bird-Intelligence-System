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
colab new -s "$SESSION" --gpu "$GPU"

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

echo "=== Step 4: Restore data from Drive ==="o

DRIVE_BACKUP_DIR = "/content/drive/MyDrive/Bird-Intelligence-System_data_and_outputs"
print(f"Drive: {os.listdir(DRIVE_BACKUP_DIR)[:5]}")

for label, rel in [("Audio archives", "archived_audio"),
                    ("Spectrograms archives", "archived_spectrograms"),
                    ("Metadata", "metadata")]:
    full = os.path.join(DRIVE_BACKUP_DIR, rel)
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
#echo "  bash scripts/colab_experiment.sh ssl_sanity --dry-run"
#echo "  bash scripts/colab_experiment.sh ssl_standard"

# Next step is to go to console then run restore
echo "  colab console -s $SESSION    # interactive shell"
echo "  colab stop -s $SESSION       # teardown when done"
