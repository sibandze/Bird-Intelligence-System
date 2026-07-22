# pipeline/pipeline_runner.py
"""Runner for data downloading and spectrogram preprocessing pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.run_pipeline import run_data_pipeline
from src.utils.configs import load_and_resolve_config


def main():
    parser = argparse.ArgumentParser(
        description="Run the Bird Intelligence System data downloading and preprocessing pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file relative to project root (default: configs/config.yaml)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 Starting Data Pipeline")
    print("=" * 80 + "\n")

    # Load and resolve paths (absolute paths + derived audio segment_size)
    resolved_config = load_and_resolve_config(ROOT_DIR, args.config)

    # Run execution
    run_data_pipeline(resolved_config)


if __name__ == "__main__":
    main()
