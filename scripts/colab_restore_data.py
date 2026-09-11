# scripts/colab_restore_data.py
"""Restore data from Google Drive into the BIS repo.

Restores spectrograms and metadata by default.
Raw audio archives are skipped unless --include-audio is passed.

Expects:
  - Google Drive already mounted at /content/drive
  - Repo already cloned at /content/Bird-Intelligence-System

Usage (from local machine):
  colab exec -s bis -f scripts/colab_restore_data.py
  colab exec -s bis -f scripts/colab_restore_data.py -- --dry-run
  colab exec -s bis -f scripts/colab_restore_data.py -- --include-audio
"""


import os
import sys
import shutil
import tarfile
import argparse
from pathlib import Path

# ── Defaults (overridable via CLI args) ─────────────────

REPO_DIR = "/content/Bird-Intelligence-System"
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/Bird-Intelligence-System_data_and_outputs"


# ── Config Loading ──────────────────────────────────────

def load_project_config(repo_dir: str) -> tuple[dict, dict]:
    """Load config.yaml and extract data paths + audio params."""
    import yaml  # available on Colab, or pip install pyyaml

    config_path = os.path.join(repo_dir, "configs", "config.yaml")
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    audio_cfg = config.get("audio", {})

    paths = {
        "raw_audio": data_cfg.get("raw_audio_dir", "data/raw_audio"),
        "spectrograms": data_cfg.get("processed_npy_dir", "data/processed_spectrograms"),
        "metadata": data_cfg.get("metadata_dir", "data/metadata"),
    }

    params = {
        "sr": audio_cfg.get("sr", 32000),
        "n_fft": audio_cfg.get("n_fft", 2048),
        "hop_length": audio_cfg.get("hop_length", 512),
        "n_mels": audio_cfg.get("n_mels", 128),
    }

    return paths, params


def build_spectrogram_pattern(params: dict) -> str:
    return f"sr{params['sr']}_nfft{params['n_fft']}_hop{params['hop_length']}_nmel{params['n_mels']}"


# ── Drive Inspection ────────────────────────────────────

def inspect_drive(backup_dir: str, max_items: int = 8):
    """Print a summary of what's in the Drive backup directory."""
    if not os.path.isdir(backup_dir):
        print(f"ERROR: Drive backup directory not found: {backup_dir}")
        print("Check that Drive is mounted and the path is correct.")
        sys.exit(1)

    contents = sorted(os.listdir(backup_dir))
    print(f"\nDrive backup: {backup_dir}")
    print(f"  {len(contents)} item(s) found:\n")

    for item in contents[:max_items]:
        item_path = os.path.join(backup_dir, item)
        if os.path.isdir(item_path):
            sub = os.listdir(item_path)
            dirs = sum(1 for s in sub if os.path.isdir(os.path.join(item_path, s)))
            files = len(sub) - dirs
            print(f"  {item}/  ({files} files, {dirs} subdirs)")
            # Show a few example files
            for sample in sorted(sub)[:3]:
                print(f"    - {sample}")
            if len(sub) > 3:
                print(f"    ... and {len(sub) - 3} more")
        else:
            size_mb = os.path.getsize(item_path) / (1024 * 1024)
            print(f"  {item}  ({size_mb:.1f} MB)")

    if len(contents) > max_items:
        print(f"\n  ... and {len(contents) - max_items} more items")


# ── Restore Functions ───────────────────────────────────

def restore_archives(
    archive_subdir: str,
    local_target_rel: str,
    file_filter=None,
    repo_dir: str = REPO_DIR,
    backup_dir: str = DRIVE_BACKUP_DIR,
    dry_run: bool = False,
) -> int:
    """Extract matching files from .tar.gz archives on Drive into repo dirs.

    Args:
        archive_subdir: Subdirectory under backup_dir containing .tar.gz files.
        local_target_rel: Relative path under repo_dir to extract into.
        file_filter: Optional callable(filename) -> bool to select files.
        dry_run: If True, only report what would be extracted.

    Returns:
        Number of files extracted (or that would be extracted).
    """
    drive_path = os.path.join(backup_dir, archive_subdir)
    local_path = os.path.join(repo_dir, local_target_rel)
    os.makedirs(local_path, exist_ok=True)

    if not os.path.isdir(drive_path):
        print(f"  [{archive_subdir}] Not found on Drive, skipping.")
        return 0

    archives = sorted(f for f in os.listdir(drive_path) if f.endswith(".tar.gz"))
    if not archives:
        print(f"  [{archive_subdir}] No .tar.gz archives found.")
        return 0

    print(f"  [{archive_subdir}] Found {len(archives)} archive(s)")
    extracted = 0

    for fname in archives:
        archive_path = os.path.join(drive_path, fname)
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                if file_filter:
                    members = [m for m in members if file_filter(m.name)]

                for member in members:
                    # Flatten: strip directory components to prevent path traversal
                    flat_name = os.path.basename(member.name)
                    dest = os.path.join(local_path, flat_name)

                    if os.path.exists(dest):
                        continue  # already restored

                    if dry_run:
                        print(f"    [DRY] Would extract: {flat_name}")
                        extracted += 1
                    else:
                        member.name = flat_name
                        tar.extract(member, path=local_path)
                        extracted += 1

        except tarfile.ReadError as e:
            print(f"    WARNING: Cannot read {fname}: {e}")
        except Exception as e:
            print(f"    WARNING: {fname}: {e}")

    action = "would extract" if dry_run else "extracted"
    print(f"    {extracted} new file(s) {action} from {len(archives)} archive(s) -> {local_target_rel}/")
    return extracted


def restore_files(
    rel_dir: str,
    file_filter=None,
    repo_dir: str = REPO_DIR,
    backup_dir: str = DRIVE_BACKUP_DIR,
    dry_run: bool = False,
) -> int:
    """Copy loose files from Drive into repo directory."""
    drive_path = os.path.join(backup_dir, rel_dir)
    local_path = os.path.join(repo_dir, rel_dir)
    os.makedirs(local_path, exist_ok=True)

    if not os.path.isdir(drive_path):
        print(f"  [{rel_dir}] Not found on Drive, skipping.")
        return 0

    candidates = sorted(os.listdir(drive_path))
    files = []
    for fname in candidates:
        full = os.path.join(drive_path, fname)
        if os.path.isfile(full) and (file_filter is None or file_filter(fname)):
            files.append(fname)

    if not files:
        print(f"  [{rel_dir}] No matching files found on Drive.")
        return 0

    print(f"  [{rel_dir}] Found {len(files)} matching file(s)")
    copied = 0

    for fname in files:
        src = os.path.join(drive_path, fname)
        dst = os.path.join(local_path, fname)

        if os.path.exists(dst):
            continue

        if dry_run:
            size_mb = os.path.getsize(src) / (1024 * 1024)
            print(f"    [DRY] Would copy: {fname} ({size_mb:.1f} MB)")
            copied += 1
        else:
            shutil.copy2(src, dst)
            copied += 1

    action = "would copy" if dry_run else "copied"
    print(f"    {copied} new file(s) {action} -> {rel_dir}/")
    return copied

# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Restore data from Google Drive into the BIS repo"
    )
    parser.add_argument("--repo-dir", default=REPO_DIR)
    parser.add_argument("--backup-dir", default=DRIVE_BACKUP_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be restored without copying")
    parser.add_argument("--include-audio", action="store_true",
                        help="Also restore raw audio archives (skipped by default)")
    parser.add_argument("--skip-spectrograms", action="store_true")
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Only show Drive contents, don't restore")
    args = parser.parse_args()

    # ── Verify prerequisites ────────────────────────────
    if not os.path.isdir("/content/drive/MyDrive"):
        print("ERROR: Google Drive not mounted at /content/drive")
        print("Run: colab drivemount -s <session>")
        sys.exit(1)

    if not os.path.isdir(args.repo_dir):
        print(f"ERROR: Repo not found at {args.repo_dir}")
        print("Clone the repo first.")
        sys.exit(1)

    # ── Load config ─────────────────────────────────────
    paths, params = load_project_config(args.repo_dir)
    spec_pattern = build_spectrogram_pattern(params)

    print(f"Repo:          {args.repo_dir}")
    print(f"Drive backup:  {args.backup_dir}")
    print(f"Audio params:  {spec_pattern}")
    print(f"Data paths:")
    for key, rel in paths.items():
        print(f"  {key:15s} -> {rel}")
    if args.dry_run:
        print("\n*** DRY RUN — no files will be modified ***")

    # ── Inspect ─────────────────────────────────────────
    inspect_drive(args.backup_dir)

    if args.inspect_only:
        return

    # ── Ensure local dirs exist ─────────────────────────
    for rel in paths.values():
        os.makedirs(os.path.join(args.repo_dir, rel), exist_ok=True)

    total = 0

    # ── 1. Audio (from .tar.gz archives) ───────────────
    # ── 1. Audio (from .tar.gz archives) ───────────────
    if args.include_audio:
        print(f"\n--- Restoring raw audio ---")
        total += restore_archives(
            "archived_audio",
            paths["raw_audio"],
            file_filter=lambda f: f.endswith((".ogg", ".wav", ".mp3")),
            repo_dir=args.repo_dir,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run,
        )
    else:
        print(f"\n--- Skipping raw audio (use --include-audio to restore) ---")

    # ── 2. Spectrograms (from .tar.gz archives, filtered by params) ──
    if not args.skip_spectrograms:
        print(f"\n--- Restoring spectrograms (pattern: {spec_pattern}) ---")
        total += restore_archives(
            "archived_spectrograms",
            paths["spectrograms"],
            file_filter=lambda f: f.endswith(".npy") and spec_pattern in f,
            repo_dir=args.repo_dir,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run,
        )

    # ── 3. Metadata (loose .csv files) ─────────────────
    if not args.skip_metadata:
        print(f"\n--- Restoring metadata (pattern: {spec_pattern}) ---")
        total += restore_files(
            paths["metadata"],
            file_filter=lambda f: f.endswith(".csv") and spec_pattern in f,
            repo_dir=args.repo_dir,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run,
        )

    # ── Summary ─────────────────────────────────────────
    action = "would be restored" if args.dry_run else "restored"
    print(f"\n{'='*60}")
    print(f"Total: {total} file(s) {action}")
    print(f"{'='*60}")

    # Show final local state
    print("\nLocal data directories:")
    for key, rel in paths.items():
        local = os.path.join(args.repo_dir, rel)
        if os.path.isdir(local):
            count = len([f for f in os.listdir(local) if os.path.isfile(os.path.join(local, f))])
            print(f"  {rel}: {count} file(s)")
        else:
            print(f"  {rel}: (does not exist)")


if __name__ == "__main__":
    main()
