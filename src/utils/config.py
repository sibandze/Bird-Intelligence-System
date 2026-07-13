from pathlib import Path


def resolve_metadata_csv_path(config):
    """Return the metadata CSV path from the absolute path provided in config."""
    data_cfg = config.get("data", {})
    metadata_dir = data_cfg.get("metadata_dir")
    
    if not metadata_dir:
        raise FileNotFoundError("No metadata directory configured.")
    
    metadata_dir_path = Path(metadata_dir)
    
    if not metadata_dir_path.exists() or not metadata_dir_path.is_dir():
        raise FileNotFoundError(f"Metadata directory does not exist: {metadata_dir_path}")
    
    csv_files = sorted(metadata_dir_path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No metadata CSV files found in {metadata_dir_path}")
    
    # If there's only one CSV, return it
    if len(csv_files) == 1:
        return str(csv_files[0])
    
    # Multiple CSVs: return the first one alphabetically
    return str(csv_files[0])
