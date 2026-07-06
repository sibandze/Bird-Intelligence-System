# src/data/process_audio.py

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

def generate_mel_spectrogram_data(audio_path, sr=32000, n_fft=2048, hop_length=512, n_mels=128):
    """
    Loads an audio file and generates its Mel spectrogram data.

    Args:
        audio_path (str): The path to the audio file.
        sr (int): Sampling rate for audio loading. Defaults to 32000.
        n_fft (int): FFT window size for Mel spectrogram. Defaults to 2048.
        hop_length (int): Number of samples between successive frames. Defaults to 512.
        n_mels (int): Number of Mel bands to generate. Defaults to 128.

    Returns:
        tuple: (numpy.ndarray, int) The Mel spectrogram in dB scale and the loaded sampling rate,
               or (None, None) if an error occurs.
    """
    try:
        # Load the audio file
        y, sr_loaded = librosa.load(audio_path, sr=sr)

        # Compute the Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr_loaded, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        # Convert to dB scale
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        return mel_spectrogram_db, sr_loaded
    except Exception as e:
        print(f"Error generating Mel spectrogram data for {audio_path}: {e}")
        return None, None


def save_spectrogram_npy(spectrogram_data, out_path):
    """
    Saves spectrogram numpy array to .npy file.

    Args:
        spectrogram_data: np.ndarray, mel spectrogram in dB
        out_path: str or Path, full path like 'processed/mels/file.npy'
    Returns: Path to saved file
    """
    np.save(out_path, spectrogram_data)
    return out_path

def preprocess_and_save(audio_path, out_path, sr=32000, n_fft=2048, hop_length=512, n_mels=128):
    """
    Load audio -> mel spectrogram -> save as .npy
    Just wraps the 2 functions above, no reimplementation.

    Returns: True if success, False if failed
    """
    mel_db, sr_loaded = generate_mel_spectrogram_data(
        audio_path, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    if mel_db is None:
        return False

    save_spectrogram_npy(mel_db, out_path)
    return True

def load_local_spectrogram(npy_path):
    """
    Load spectrogram.npy from disk.
    Returns: np.ndarray shape (n_mels, T)
    Raises FileNotFoundError if path invalid.
    """
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"Spectrogram not found: {npy_path}")
    return np.load(npy_path)

def visualize_mel_spectrogram(spectrogram_data, sr, title='Mel Spectrogram', hop_length=512):
    """
    Visualizes a Mel spectrogram numpy array.

    Args:
        spectrogram_data (numpy.ndarray): The Mel spectrogram data in dB.
        sr (int): Sampling rate.
        title (str): Title for the plot. Defaults to 'Mel Spectrogram'.
        hop_length (int): Number of samples between successive frames.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_data, sr=sr, x_axis='time', y_axis='mel', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def save_spectrogram_image(spectrogram_data, sr, output_path, hop_length=512):
    """
    Saves a Mel spectrogram numpy array as an image.

    Args:
        spectrogram_data (numpy.ndarray): The Mel spectrogram data in dB.
        sr (int): Sampling rate.
        output_path (str): The full path to save the spectrogram image.
        hop_length (int): Number of samples between successive frames.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_data, sr=sr, x_axis='time', y_axis='mel', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
