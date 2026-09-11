#!/usr/bin/env bash
# scripts/colab_file_ops.sh
set -euo pipefail

SESSION="${COLAB_SESSION:-bis}"
REPO_DIR="/content/Bird-Intelligence-System"
ACTION="${1:?Usage: $0 {upload-config|download-results|download-checkpoint} [args...]}"

case "$ACTION" in
    upload-config)
        LOCAL_CFG="${2:?Local config path}"
        colab upload -s "$SESSION" "$LOCAL_CFG" "$REPO_DIR/configs/config.yaml"
        echo "Config uploaded."
        ;;

    download-results)
        LOCAL_DIR="${2:-./results}"
        mkdir -p "$LOCAL_DIR"
        # List what's available
        echo "Remote results:"
        colab ls -s "$SESSION" "$REPO_DIR/results/"
        read -rp "Enter experiment dir name to download: " EXP_NAME
        colab download -s "$SESSION" "$REPO_DIR/results/$EXP_NAME" "$LOCAL_DIR/$EXP_NAME"
        echo "Downloaded to $LOCAL_DIR/$EXP_NAME"
        ;;

    download-checkpoint)
        RUN_DIR="${2:?Run directory name (e.g. run_0000_baseline_lr_sweep)}"
        LOCAL_DIR="${3:-./checkpoints}"
        mkdir -p "$LOCAL_DIR"
        colab download -s "$SESSION" \
            "$REPO_DIR/results/"*"/$RUN_DIR/best_model.pth" \
            "$LOCAL_DIR/$RUN_DIR-best_model.pth" 2>/dev/null \
        || colab download -s "$SESSION" \
            "$REPO_DIR/results/"*"/$RUN_DIR/checkpoint_best.pth" \
            "$LOCAL_DIR/$RUN_DIR-checkpoint_best.pth"
        echo "Checkpoint downloaded to $LOCAL_DIR/"
        ;;

    *)
        echo "Unknown action: $ACTION"
        echo "Available: upload-config, download-results, download-checkpoint"
        exit 1
        ;;
esac
