#!/usr/bin/env bash
# script/colab_experiments.sh
set -euo pipefail

SUITE="${1:?Usage: $0 <suite> [--dry-run]}"
SESSION="${COLAB_SESSION:-bis}"
REPO_DIR="/content/Bird-Intelligence-System"
DRY_RUN=""
MODE="auto"

# Parse remaining args
shift
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        --mode=*)  MODE="${arg#*=}" ;;
    esac
done

# Infer mode from suite name
if [ "$MODE" = "auto" ]; then
    case "$SUITE" in
        ssl_*) MODE="ssl" ;;
        *)     MODE="supervised" ;;
    esac
fi

echo "=== Running: suite=$SUITE mode=$MODE session=$SESSION ==="

colab exec -s "$SESSION" <<PYEOF
import subprocess, sys, os

cmd = [
    sys.executable, "-m", "experiments.experiment_runner",
    "--suite", "$SUITE",
    "--mode", "$MODE",
    "--config", "configs/config.yaml",
    "--seed", "42",
]
if "$DRY_RUN":
    cmd.append("--dry-run")

print(f"Executing: {' '.join(cmd)}\n")
result = subprocess.run(cmd, cwd="$REPO_DIR")
sys.exit(result.returncode)
PYEOF

echo "=== Done ==="
