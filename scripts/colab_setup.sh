#!/usr/bin/env bash
# scripts/colab_setup.sh
set -euo pipefail

SESSION="${1:-bis}"
REPO_URL="https://github.com/sibandze/Bird-Intelligence-System.git"
BRANCH="dev-unsupervised"
REPO_DIR="/content/Bird-Intelligence-System"

# Order to try - Colab T4 is most contested
GPU_CANDIDATES=("T4" "L4" "A100" "V100")

# If user passed a GPU as $2, try it first
if [[ $# -ge 2 ]]; then
  GPU_CANDIDATES=("$2" "${GPU_CANDIDATES[@]}")
fi

echo "=== Step 1: Provision VM (with fallback) ==="

# Clean up any existing session with same name to avoid 412 TooManyAssignments
colab stop -s "$SESSION" 2>/dev/null || true

ASSIGNED_GPU=""
for gpu in "${GPU_CANDIDATES[@]}"; do
  echo "[colab] Trying accelerator: $gpu..."
  if colab new -s "$SESSION" --gpu "$gpu" 2>&1; then
    ASSIGNED_GPU="$gpu"
    echo "[colab] Assigned $gpu"
    break
  else
    echo "[colab] $gpu unavailable, trying next..."
    colab stop -s "$SESSION" 2>/dev/null || true
    sleep 2
  fi
done

if [[ -z "$ASSIGNED_GPU" ]]; then
  echo "[colab] All GPU types failed - falling back to CPU runtime"
  if ! colab new -s "$SESSION" 2>&1; then
    # Some colab-cli versions use --variant CPU or no flag
    echo "[colab] Retrying CPU as --variant CPU..."
    colab new -s "$SESSION" --variant CPU 2>&1 || {
      echo "ERROR: Could not provision any runtime (GPU or CPU). Colab may be down."
      exit 1
    }
  fi
  ASSIGNED_GPU="CPU"
fi

echo "=== Step 2: Mount Google Drive ==="
colab drivemount -s "$SESSION" || echo "[warn] drivemount failed - continuing"

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

echo "=== Step 4: Inspect data from Drive ==="
colab exec -s "$SESSION" <<PYEOF
import os
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/Bird-Intelligence-System_data_and_outputs"
if os.path.exists(DRIVE_BACKUP_DIR):
    print(f"Drive: {os.listdir(DRIVE_BACKUP_DIR)[:20]}")
    for label, rel in [("Audio archives", "archived_audio"),
                        ("Spectrograms archives", "archived_spectrograms"),
                        ("Metadata", "metadata")]:
        full = os.path.join(DRIVE_BACKUP_DIR, rel)
        if os.path.isdir(full):
            n = len([f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))])
            print(f"  {label}: {n} file(s)")
        else:
            print(f"  {label}: missing ({full})")
else:
    print(f"WARNING: Drive dir not found: {DRIVE_BACKUP_DIR}")
PYEOF

echo ""
echo "=== Setup complete ==="
echo "Session: $SESSION | Accelerator: $ASSIGNED_GPU"
echo ""
echo "Next steps:"
echo "  colab console -s $SESSION"
echo "  colab stop -s $SESSION"
