# src/data/run_pipeline.py

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from .download import download_audio
from .process_audio import preprocess_and_save


def get_spectrogram_frames(npy_path: str) -> int:
    """
    Get total number of time frames from a saved spectrogram.

    Loads only the shape from the .npy file header without loading
    the full array into memory when using mmap_mode='r'.

    Args:
        npy_path: Path to .npy spectrogram file

    Returns:
        Total number of time frames (T dimension)
    """
    try:
        # Memory-map the file to read shape without loading full array
        mel = np.load(npy_path, mmap_mode='r')
        return mel.shape[1]  # T dimension: [n_mels, T]
    except Exception:
        return 0


def run_data_pipeline(config, use_full_dataset: bool = False):
    start_time = time.time()
    data_cfg = config["data"]
    audio_cfg = config["audio"]

    RAW_AUDIO_DIR = Path(data_cfg["raw_audio_dir"])
    PROCESSED_NPY_DIR = Path(data_cfg["processed_npy_dir"])
    num_classes = data_cfg.get("num_classes")
    num_samples_per_class = data_cfg.get("num_samples_per_class")
    segment_size = audio_cfg["segment_size"]

    os.makedirs(RAW_AUDIO_DIR, exist_ok=True)
    os.makedirs(PROCESSED_NPY_DIR, exist_ok=True)

    print("==================================================")
    print("Starting Data Pipeline")
    print("==================================================")

    print("\nLoading metadata...")
    df = pd.read_csv(data_cfg["data_csv"])

    df = df[["common_name", "scientific_name", "Download_link", "xc_id"]].copy()
    df = df.dropna(
        subset=["Download_link", "xc_id", "scientific_name"]
    ).reset_index(drop=True)
    
    # Check config or argument override for full dataset mode
    if use_full_dataset or data_cfg.get("use_full_dataset", False):
        print("🌐 Mode: Full Dataset (No balancing/class filtering)")
        df_target = df.copy()
    else:
        print(f"⚖️ Mode: Balanced Dataset (Top {num_classes} classes, {num_samples_per_class} samples/class)")
        top_classes = (
            df["scientific_name"].value_counts().head(num_classes).index.tolist()
        )
        
        # Print a table of the top species and their total recording counts
        print(f"\nTop {num_classes} Species by Recording Count (source metadata):")
        print("-" * 80)
        print(f"{'Rank':<5} {'Common Name':<30} {'Scientific Name':<30} {'Recordings':>10}")
        print("-" * 80)

        top_counts = df["scientific_name"].value_counts().head(num_classes)
        for rank, (sci_name, count) in enumerate(top_counts.items(), start=1):
            # Get the first common name associated with this scientific name
            common_name = df.loc[df["scientific_name"] == sci_name, "common_name"].iloc[0]
            print(f"{rank:<5} {common_name:<30} {sci_name:<30} {count:>10,}")

        print("-" * 80)
        
        df_target = pd.DataFrame()
        for sci_name in top_classes:
            class_samples = df[df["scientific_name"] == sci_name].head(
                num_samples_per_class
            )
            df_target = pd.concat([df_target, class_samples])

    df_target = df_target.reset_index(drop=True)

    # Map class IDs dynamically
    unique_sci = df_target["scientific_name"].unique()
    sci_to_id = {name: i for i, name in enumerate(unique_sci)}
    df_target["scientific_name_id"] = df_target["scientific_name"].map(sci_to_id)

    total_requested = len(df_target)

    print("\nMetadata summary")
    print("----------------")
    print(f"Total rows in source : {len(df):,}")
    print(f"Selected classes     : {len(unique_sci):,}")
    print(f"Target total samples : {total_requested:,}")

    processed_rows = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    print("\nDownloading and preprocessing audio...")
    pbar = tqdm(
        df_target.iterrows(),
        total=total_requested,
        desc="Processing audio",
        unit="file",
    )

    for _, row in pbar:
        rc_id = str(row["xc_id"])
        url = row["Download_link"]

        audio_filename = f"{rc_id}.ogg"
        npy_filename = (
            f"{rc_id}_sr{audio_cfg['sr']}_nfft{audio_cfg['n_fft']}"
            f"_hop{audio_cfg['hop_length']}_nmel{audio_cfg['n_mels']}"
            f"_seg{segment_size}.npy"
        )

        local_audio_path = RAW_AUDIO_DIR / audio_filename
        local_npy_path = PROCESSED_NPY_DIR / npy_filename

        # Step A: Spectrogram cached check
        if local_npy_path.exists():
            # Get total frames from cached spectrogram
            total_frames = get_spectrogram_frames(str(local_npy_path))

            if total_frames > 0:
                row_dict = row.to_dict()
                row_dict["rc_id"] = xc_id
                row_dict["scientific_name_id"] = sci_to_id[row["scientific_name"]]
                row_dict["spectrogram_filename"] = npy_filename
                row_dict["local_spectrogram_path"] = str(local_npy_path)
                row_dict["total_frames"] = total_frames  # Store frame count
                processed_rows.append(row_dict)

                skipped_count += 1
                pbar.set_postfix(
                    proc=processed_count, skip=skipped_count, fail=failed_count
                )
                continue 
            else:
                tqdm.write(f"⚠️ Corrupted cached spectrogram for {rc_id}, reprocessing...")
                # Fall through to reprocess
                pass

        # Step B: Download audio if missing
        if not local_audio_path.exists():
            downloaded_file = download_audio(
                url, audio_filename, output_dir=str(RAW_AUDIO_DIR)
            )
            if not downloaded_file:
                tqdm.write(f"⚠️ Failed downloading {xc_id}")
                failed_count += 1
                pbar.set_postfix(
                    proc=processed_count, skip=skipped_count, fail=failed_count
                )
                continue

        # Step C: Preprocess audio to NPY spectrogram
        # Only run if we didn't already process from cache
        if not local_npy_path.exists():
            success = preprocess_and_save(
                str(local_audio_path),
                str(local_npy_path),
                sr=audio_cfg["sr"],
                n_fft=audio_cfg["n_fft"],
                hop_length=audio_cfg["hop_length"],
                n_mels=audio_cfg["n_mels"],
            )
        else:
            # Re-process if previous cache was corrupted
            success = preprocess_and_save(
                str(local_audio_path),
                str(local_npy_path),
                sr=audio_cfg["sr"],
                n_fft=audio_cfg["n_fft"],
                hop_length=audio_cfg["hop_length"],
                n_mels=audio_cfg["n_mels"],
            )

        if success:
            # Get total frames from newly created spectrogram
            total_frames = get_spectrogram_frames(str(local_npy_path))

            row_dict = row.to_dict()
            row_dict["scientific_name_id"] = sci_to_id[row["scientific_name"]]
            row_dict["spectrogram_filename"] = npy_filename
            row_dict["local_spectrogram_path"] = str(local_npy_path)
            row_dict["total_frames"] = total_frames  # Store frame count
            row_dict["rc_id"] = rc_id
            processed_rows.append(row_dict)

            processed_count += 1
        else:
            tqdm.write(f"⚠️ Could not create spectrogram for {xc_id}")
            failed_count += 1

        pbar.set_postfix(
            proc=processed_count, skip=skipped_count, fail=failed_count
        )

    # Add summary statistics to final DataFrame
    if processed_rows:
        # Output aligned metadata CSV
        tag = "full" if (use_full_dataset or data_cfg.get("use_full_dataset", False)) else "balanced"
        output_metadata_csv = Path(data_cfg["metadata_dir"]) / (
            f"metadata_{tag}_sr{audio_cfg['sr']}_nfft{audio_cfg['n_fft']}"
            f"_hop{audio_cfg['hop_length']}_nmel{audio_cfg['n_mels']}"
            f"_seg{segment_size}.csv"
        )

        os.makedirs(output_metadata_csv.parent, exist_ok=True)
        final_df = pd.DataFrame(processed_rows)

        # Add derived column: number of valid windows per recording
        # This is useful for the sliding window strategy
        window_stride = config.get("window", {}).get("stride",
                                                     audio_cfg.get("window_stride", 256))

        def calculate_num_windows(total_frames: int, segment_size: int, stride: int) -> int:
            """Match SlidingWindowStrategy._build_starts including forced tail."""
            if total_frames <= segment_size:
                return 1
            max_start = total_frames - segment_size
            # Number of windows from range(0, max_start+1, stride)
            base = (max_start // stride) + 1
            # Add the forced tail window if max_start is not a multiple of stride
            if max_start % stride != 0:
                base += 1
            return base

        final_df["num_windows"] = final_df["total_frames"].apply(
            lambda frames: calculate_num_windows(frames, segment_size, window_stride)
        )

        final_df.to_csv(output_metadata_csv, index=False)
        # Print all rows of the final metadata for inspection
        dups = set()
        num_dups = 0
        print("\n--- Final metadata rows ---")
        for idx, row in final_df.iterrows():
            if row["rc_id"] not in dups:
                dups.add(row["rc_id"])
            else:
                num_dups+=1
            print(f"Row {idx}:")
            for col in final_df.columns:
                print(f"  {col}: {row[col]}")
            print("-" * 60)
        print("Duplicates:", num_dups)
        elapsed_time = time.time() - start_time
        mins, secs = divmod(int(elapsed_time), 60)
        hrs, mins = divmod(mins, 60)
        time_str = f"{hrs}h {mins}m {secs}s" if hrs > 0 else f"{mins}m {secs}s"

        # Calculate frame statistics
        mean_frames = final_df["total_frames"].mean()
        min_frames = final_df["total_frames"].min()
        max_frames = final_df["total_frames"].max()

        print("\n==================================================")
        print("Pipeline Complete")
        print("==================================================")
        print(f"Dataset mode         : {tag.upper()}")
        print(f"Classes              : {len(unique_sci):,}")
        print(f"Requested samples    : {total_requested:,}")
        print(f"Processed            : {processed_count:,}")
        print(f"Skipped (cached)     : {skipped_count:,}")
        print(f"Failed               : {failed_count:,}")
        print(f"Total retained       : {len(final_df):,}")
        print(f"\nFrame statistics:")
        print(f"  Mean frames        : {mean_frames:.1f}")
        print(f"  Min frames         : {min_frames}")
        print(f"  Max frames         : {max_frames}")
        print(f"  Segment size       : {segment_size}")
        print(f"  Mean windows       : {final_df['num_windows'].mean():.1f}")
        print(f"\nMetadata CSV         : {output_metadata_csv.name}")
        print(f"Raw audio retained   : {RAW_AUDIO_DIR}")
        print(f"Elapsed time         : {time_str}")
        print("==================================================\n")
    else:
        print("\n⚠️ No samples were successfully processed!")
