# src/data/run_pipeline.py

import os
import pandas as pd
from pathlib import Path

from .download import download_audio
from .process_audio import preprocess_and_save

def run_data_pipeline(config):
    data_cfg = config['data']
    audio_cfg = config['audio']

    RAW_AUDIO_DIR = Path(data_cfg['raw_audio_dir'])
    PROCESSED_NPY_DIR = Path(data_cfg['processed_npy_dir'])
    num_classes = data_cfg['num_classes']
    num_samples_per_class = data_cfg['num_samples_per_class']

    os.makedirs(RAW_AUDIO_DIR, exist_ok=True)
    os.makedirs(PROCESSED_NPY_DIR, exist_ok=True)

    print("Step 1: Downloading and reading voices metadata...")
    df = pd.read_csv(data_cfg['data_csv'])

    df = df[['common_name', 'scientific_name', 'Download_link', 'xc_id']].copy()
    df = df.dropna(subset=['Download_link', 'xc_id', 'scientific_name']).reset_index(drop=True)

    print(f"Step 2: Balancing dataset (Top {num_classes} classes, {num_samples_per_class} samples each)...")
    top_classes = df['scientific_name'].value_counts().head(num_classes).index.tolist()

    df_balanced = pd.DataFrame()
    for sci_name in top_classes:
        class_samples = df[df['scientific_name'] == sci_name].head(num_samples_per_class)
        df_balanced = pd.concat([df_balanced, class_samples])

    df_balanced = df_balanced.reset_index(drop=True)

    unique_sci = df_balanced['scientific_name'].unique()
    sci_to_id = {name: i for i, name in enumerate(unique_sci)}
    df_balanced['scientific_name_id'] = df_balanced['scientific_name'].map(sci_to_id)

    processed_rows = []

    print("\nStep 3: Beginning Download and Spectrogram Processing loop...")
    for idx, row in df_balanced.iterrows():
        xc_id = str(row['xc_id'])
        url = row['Download_link']

        audio_filename = f"{xc_id}.ogg"
        npy_filename = (
            f"{xc_id}_sr{audio_cfg['sr']}_nfft{audio_cfg['n_fft']}"
            f"_hop{audio_cfg['hop_length']}_nmel{audio_cfg['n_mels']}"
            f"_seg{audio_cfg['segment_size']}.npy"
        )

        local_audio_path = RAW_AUDIO_DIR / audio_filename
        local_npy_path = PROCESSED_NPY_DIR / npy_filename

        print(f"[{idx+1}/{len(df_balanced)}] Processing {xc_id} ({row['scientific_name']})...")

        if local_npy_path.exists():
            print("  -> Spectrogram already exists. Skipping download.")
            row_dict = row.to_dict()
            row_dict['scientific_name_id'] = sci_to_id[row['scientific_name']]
            row_dict['local_spectrogram_path'] = str(local_npy_path)
            processed_rows.append(row_dict)
            continue

        if not local_audio_path.exists():
            downloaded_file = download_audio(url, audio_filename, output_dir=str(RAW_AUDIO_DIR))
            if not downloaded_file:
                print(f"  ⚠️ Failed downloading {xc_id}. Skipping.")
                continue

        success = preprocess_and_save(
            str(local_audio_path),
            str(local_npy_path),
            sr=audio_cfg['sr'],
            n_fft=audio_cfg['n_fft'],
            hop_length=audio_cfg['hop_length'],
            n_mels=audio_cfg['n_mels']
        )

        if success:
            row_dict = row.to_dict()
            row_dict['scientific_name_id'] = sci_to_id[row['scientific_name']]
            row_dict['spectrogram_filename'] = npy_filename
            row_dict['local_spectrogram_path'] = str(local_npy_path)
            processed_rows.append(row_dict)
        else:
            print(f"  ⚠️ Spectrogram processing failed for {xc_id}.")

    output_metadata_csv = Path(data_cfg['metadata_dir']) / (
        f"metadata_sr{audio_cfg['sr']}_nfft{audio_cfg['n_fft']}"
        f"_hop{audio_cfg['hop_length']}_nmel{audio_cfg['n_mels']}"
        f"_seg{audio_cfg['segment_size']}.csv"
    )

    os.makedirs(output_metadata_csv.parent, exist_ok=True)

    final_df = pd.DataFrame(processed_rows)
    final_df.to_csv(output_metadata_csv, index=False)

    print("\nStep 4: Cleaning up raw audio files...")
    deleted_count = 0
    for audio_file in RAW_AUDIO_DIR.glob("*.ogg"):
        try:
            audio_file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"  ⚠️ Failed to delete {audio_file}: {e}")

    print("\n" + "="*50)
    print(f"Processing complete! Aligned dataset metadata saved to: {output_metadata_csv}")
    print(f"Successfully processed {len(final_df)} files across {num_classes} classes.")
    print(f"Cleaned up {deleted_count} raw audio files.")
    print("="*50)
