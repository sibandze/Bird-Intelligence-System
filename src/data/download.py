# src/data/download.py

import os
import requests

def download_audio(url, filename, output_dir):
    """
    Downloads an audio file from a given URL.

    Args:
        url (str): The URL of the audio file.
        filename (str): The desired filename for the downloaded audio.
        output_dir (str): The directory to save the downloaded audio file.

    Returns:
        str: The full path to the downloaded file, or None if download fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {filepath}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
