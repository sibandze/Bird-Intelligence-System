#!/usr/bin/env bash
# scripts/colab_fresh_run.sh
set -euo pipefail

# Uses 'colab run' for a fire-and-forget experiment.
# Provisions a VM, executes setup+run, downloads results, tears down.

SUITE="${1:?Usage: $0 <suite> [gpu]}"
GPU="${2:-T4}"
REPO_DIR="/content/Bird-Intelligence-System"

# Create a temporary all-in-one script
TMPFILE=$(mktemp /tmp/colab_run_XXXXXX.py)
cat > "$TMPFILE" << 'PYEOF'
import subprocess, sys, os, shutil

REPO = "/content/Bird-Intelligence-System"
BRANCH = "dev-unsupervised"
SUITE = os.environ.get("SUITE", "ssl_sanity")
MODE = os.environ.get("MODE", "auto")

# Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Clone
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-b", BRANCH,
        "https://github.com/sibandze/Bird-Intelligence-System.git", REPO],
        check=True)
else:
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=REPO, check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=REPO, check=True)

# Install
subprocess.run([sys.executable, "-m", "pip", "install", "-r",
    os.path.join(REPO, "requirements.txt")], check=True)

# Restore data from Drive
restore_script = os.path.join(REPO, "scripts", "colab_restore_data.py")
if os.path.exists(restore_script):
    subprocess.run([sys.executable, restore_script], check=True)
else:
    print("WARNING: colab_restore_data.py not found, skipping data restore")
    print("Data must be available for experiments to run.")

# Run experiment
cmd = [sys.executable, "-m", "experiments.experiment_runner",
    "--suite", SUITE, "--mode", MODE,
    "--config", "configs/config.yaml", "--seed", "42"]
print(f"\nRunning: {' '.join(cmd)}\n")

result = subprocess.run(cmd, cwd=REPO)
sys.exit(result.returncode)
PYEOF

# Determine mode
MODE="supervised"
[[ "$SUITE" == ssl_* ]] && MODE="ssl"

echo "=== Ephemeral run: suite=$SUITE gpu=$GPU mode=$MODE ==="
echo "This will provision, run, and tear down automatically."

SUITE="$SUITE" MODE="$MODE" colab run --gpu "$GPU" --high-mem "$TMPFILE"
rm -f "$TMPFILE"
