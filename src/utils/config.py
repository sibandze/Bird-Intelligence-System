from pathlib import Path


def resolve_metadata_csv_path(config):
    """Return the metadata CSV path that matches the audio configuration."""
    data_cfg = config.get("data", {})
    metadata_dir = data_cfg.get("metadata_dir")
    
    if not metadata_dir:
        raise FileNotFoundError("No metadata directory configured.")
    
    metadata_dir_path = Path(metadata_dir)
    
    if not metadata_dir_path.exists() or not metadata_dir_path.is_dir():
        # Create the directory if it doesn't exist
        metadata_dir_path.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(f"Metadata directory did not exist (created at {metadata_dir_path}). Please add the metadata CSV file with the audio configuration specification.")
    
    csv_files = sorted(metadata_dir_path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No metadata CSV files found in {metadata_dir_path}")


    # Check for the file matching audio configuration values (sr, nfft, hop, nmel, seg)
    # If there's only one CSV, return it
    if len(csv_files) == 1:
        return str(csv_files[0])
    
    # Multiple CSVs: find the one matching the audio config
    audio_cfg = config.get("audio", {})
    signature_parts = []
    if audio_cfg.get("sr") is not None:
        signature_parts.append(f"sr{audio_cfg['sr']}")
    if audio_cfg.get("n_fft") is not None:
        signature_parts.append(f"nfft{audio_cfg['n_fft']}")
    if audio_cfg.get("hop_length") is not None:
        signature_parts.append(f"hop{audio_cfg['hop_length']}")
    if audio_cfg.get("n_mels") is not None:
        signature_parts.append(f"nmel{audio_cfg['n_mels']}")
    if audio_cfg.get("segment_size") is not None:
        signature_parts.append(f"seg{audio_cfg['segment_size']}")
    
    # Find CSV that matches all audio config parameters
    for csv_path in csv_files:
        name = csv_path.stem.lower()
        if all(part.lower() in name for part in signature_parts):
            return str(csv_path)
    
    # If no exact match found, raise error
    raise FileNotFoundError(
        f"No metadata CSV matched the audio configuration (sr={audio_cfg.get('sr')}, "
        f"n_fft={audio_cfg.get('n_fft')}, hop_length={audio_cfg.get('hop_length')}, "
        f"n_mels={audio_cfg.get('n_mels')}, segment_size={audio_cfg.get('segment_size')}) in {metadata_dir_path}. "
        f"Available files: {[path.name for path in csv_files]}"
    )
