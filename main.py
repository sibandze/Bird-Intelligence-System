import argparse
import yaml
from pathlib import Path

from src.data.run_pipeline import run_data_pipeline
# from src.training.train import train_model       # Uncomment when built
# from src.evaluation.evaluate import evaluate_model # Uncomment when built

 # Determine the absolute path of the project root (where main.py is located) and define the absolute path to the config file
ROOT_DIR = Path(__file__).resolve().parent
CONFIG_FILE_PATH = "configs/config.yaml"

def load_and_resolve_config(config_path):
    """Loads the YAML config and resolves all relative paths to absolute paths."""
    # 1. Load the YAML file
    full_config_path = ROOT_DIR / config_path
    with open(full_config_path, "r") as file:
        config = yaml.safe_load(file)

    # 3. Inject the root directory into the config for global awareness
    config['project_root'] = str(ROOT_DIR)

    # 4. Resolve data paths
    if 'data' in config:
        data_cfg = config['data']
        # Convert relative string paths to absolute resolved strings
        data_cfg['data_csv'] = str(ROOT_DIR / data_cfg['data_csv'])
        data_cfg['raw_audio_dir'] = str(ROOT_DIR / data_cfg['raw_audio_dir'])
        data_cfg['processed_npy_dir'] = str(ROOT_DIR / data_cfg['processed_npy_dir'])
        data_cfg['metadata_csv'] = str(ROOT_DIR / data_cfg['metadata_csv'])
    return config

def main():
    parser = argparse.ArgumentParser(description="Bird Intelligence System Orchestrator")
    parser.add_argument("--config", type=str, default=CONFIG_FILE_PATH, help="Path to the config file")
    parser.add_argument("--pipeline", action="store_true", help="Run the data downloading and preprocessing pipeline")
    parser.add_argument("--train", action="store_true", help="Run the end-to-end training script")
    parser.add_argument("--evaluate", action="store_true", help="Run the evaluation and analysis script")
    parser.add_argument("--all", action="store_true", help="Run the entire workflow sequentially")

    args = parser.parse_args()

    # Load and resolve all paths
    config = load_and_resolve_config(args.config)

    # 1. Data Pipeline
    if args.pipeline or args.all:
        print(">>> Starting Data Pipeline...")
        run_data_pipeline(config)

    # 2. Training Loop
    if args.train or args.all:
        print(">>> Starting Model Training...")
        # train_model(config)

    # 3. Evaluation
    if args.evaluate or args.all:
        print(">>> Starting Evaluation and Analysis...")
        # evaluate_model(config)

    if not (args.pipeline or args.train or args.evaluate or args.all):
        print("No action specified. Use --help to see available arguments.")

if __name__ == "__main__":
    main()
