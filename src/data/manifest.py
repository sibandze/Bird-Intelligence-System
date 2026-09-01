# src/data/manifest.py
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import threading

# Thread-safe lock for manifest writes
_lock = threading.Lock()


def get_rc_id(download_link: str) -> str:
    """
    Generate stable, dataset-agnostic recording id from Download_link.
    Same link = same rc_id forever, no clash across datasets.
    """
    # Normalize link: strip whitespace, lower not safe for urls so only strip
    clean = download_link.strip()
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]


def _default_manifest_path(config: Optional[Dict[str, Any]] = None) -> Path:
    if config and "data" in config and "metadata_dir" in config["data"]:
        return Path(config["data"]["metadata_dir"]) / "manifest.json"
    return Path("data/metadata/manifest.json")


def load_manifest(
    manifest_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    path = Path(manifest_path) if manifest_path else _default_manifest_path(config)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # backup corrupted file
        if path.exists():
            corrupt_backup = path.with_suffix(f".corrupt_{int(time.time())}.json")
            path.rename(corrupt_backup)
        return {}


def save_manifest(
    manifest: Dict[str, Any],
    manifest_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
):
    path = Path(manifest_path) if manifest_path else _default_manifest_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        tmp = path.with_suffix(".tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        tmp.replace(path)


def get_entry(rc_id: str, manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return manifest.get(rc_id)


def is_processed(
    rc_id: str, manifest: Dict[str, Any], npy_path: Optional[Path] = None
) -> bool:
    """Check if entry exists and spectrogram file exists."""
    entry = manifest.get(rc_id)
    if not entry:
        return False
    if not entry.get("processed"):
        return False
    if npy_path and not Path(npy_path).exists():
        return False
    # also check if file path in entry exists
    spec_path = entry.get("spectrogram_path")
    if spec_path and not Path(spec_path).exists():
        return False
    return True


def is_downloaded(
    rc_id: str, manifest: Dict[str, Any], audio_path: Optional[Path] = None
) -> bool:
    entry = manifest.get(rc_id)
    if audio_path and Path(audio_path).exists():
        return True
    if not entry:
        return False
    raw_path = entry.get("raw_audio_path")
    if raw_path and Path(raw_path).exists():
        return True
    return False


def upsert_entry(
    manifest: Dict[str, Any],
    rc_id: str,
    download_link: str,
    common_name: str,
    scientific_name: str,
    raw_audio_path: str,
    spectrogram_path: Optional[str] = None,
    original_id: Optional[str] = None,
    source: str = "xeno-canto",
    total_frames: Optional[int] = None,
    processed: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create or update entry. Returns updated manifest."""
    now = time.time()
    entry = manifest.get(rc_id, {})

    entry.update(
        {
            "rc_id": rc_id,
            "download_link": download_link,
            "common_name": common_name,
            "scientific_name": scientific_name,
            "raw_audio_path": raw_audio_path,
            "source": source,
            "updated_at": now,
        }
    )
    if original_id:
        entry["original_id"] = str(original_id)
    if spectrogram_path:
        entry["spectrogram_path"] = str(spectrogram_path)
    if total_frames is not None:
        entry["total_frames"] = int(total_frames)
    if processed:
        entry["processed"] = True
        entry["processed_at"] = now
    if "created_at" not in entry:
        entry["created_at"] = now
    if extra:
        entry.update(extra)

    manifest[rc_id] = entry
    return manifest


def get_or_create_rc_id(
    row: Dict[str, Any],
    manifest: Dict[str, Any],
    raw_audio_dir: Path,
    processed_npy_dir: Path,
    audio_cfg: Dict[str, Any],
    segment_size: int,
) -> tuple[str, Path, Path]:
    """
    From a dataframe row, get stable rc_id and expected file paths.
    """
    url = row["Download_link"]
    rc_id = get_rc_id(url)

    audio_filename = f"{rc_id}.ogg"
    npy_filename = (
        f"{rc_id}_sr{audio_cfg['sr']}_nfft{audio_cfg['n_fft']}"
        f"_hop{audio_cfg['hop_length']}_nmel{audio_cfg['n_mels']}"
        f"_seg{segment_size}.npy"
    )

    return rc_id, raw_audio_dir / audio_filename, processed_npy_dir / npy_filename


def manifest_to_dataframe(manifest: Dict[str, Any]) -> "pd.DataFrame":
    """Convert manifest to dataframe for metadata CSV export."""
    import pandas as pd

    if not manifest:
        return pd.DataFrame()
    df = pd.DataFrame(list(manifest.values()))
    return df
