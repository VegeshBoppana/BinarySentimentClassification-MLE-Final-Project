import os
import zipfile
import requests
import shutil

TRAIN_URL = "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_train_dataset.zip"
INFERENCE_URL = "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_inference_dataset.zip"

def download_and_extract(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    zip_path = os.path.join(save_dir, "data.zip")

    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(save_dir)
    os.remove(zip_path)

    # Move CSV files to data/raw/ directly because the downloading is happening in a nested folder kind of way
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if file.endswith('.csv'):
                src = os.path.join(root, file)
                dst = os.path.join(save_dir, file)
                if src != dst:
                    shutil.move(src, dst)
                    print(f"Moved {file} to {save_dir}")

    # Remove empty folders
    for item in os.listdir(save_dir):
        item_path = os.path.join(save_dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Removed folder {item}")

    print(f"Done! Data saved to {save_dir}")

if __name__ == "__main__":
    print("=== Downloading Train Data ===")
    download_and_extract(TRAIN_URL, 'data/raw')

    print("\n=== Downloading Inference Data ===")
    download_and_extract(INFERENCE_URL, 'data/raw')

    print("\nAll data downloaded successfully!")