from pathlib import Path


def resolve_metadata_csv_path(config):
    """Resolve the metadata CSV that matches the current spectrogram configuration."""
    data_cfg = config.get("data", {})

    metadata_dir = data_cfg.get("metadata_dir")
    if metadata_dir:
        metadata_dir_path = Path(metadata_dir)
        if not metadata_dir_path.is_absolute():
            project_root = config.get("project_root")
            if project_root:
                metadata_dir_path = Path(project_root) / metadata_dir_path
            else:
                metadata_dir_path = metadata_dir_path.resolve()

        if metadata_dir_path.exists() and metadata_dir_path.is_dir():
            csv_files = sorted(metadata_dir_path.glob("*.csv"))
            if len(csv_files) == 1:
                return str(csv_files[0])

            if csv_files:
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

                matches = []
                for csv_path in csv_files:
                    name = csv_path.stem.lower()
                    score = sum(1 for part in signature_parts if part.lower() in name)
                    if score:
                        matches.append((score, csv_path))

                if matches:
                    best_score = max(score for score, _ in matches)
                    best_matches = [csv_path for score, csv_path in matches if score == best_score]
                    if len(best_matches) == 1:
                        return str(best_matches[0])

                    best_matches.sort(key=lambda path: path.name)
                    return str(best_matches[0])

            raise FileNotFoundError(
                f"No metadata CSV matched the current configuration in {metadata_dir_path}. "
                f"Available files: {[path.name for path in csv_files]}"
            )

    legacy_csv = data_cfg.get("metadata_csv")
    if legacy_csv:
        return str(Path(legacy_csv))

    raise FileNotFoundError("No metadata directory or metadata CSV was configured.")
