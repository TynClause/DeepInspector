import os
import requests
import zipfile
import argparse

dataset_urls = {
    'african-wildlife': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip',
}

def download_and_unzip(url, extract_to="."):
    """Download and unzip a file from a URL.

    Args:
        url (str): The URL to download the file from.
        extract_to (str): The directory to extract the contents to.
    """
    os.makedirs(extract_to, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = os.path.join(extract_to, filename)

    print(f"Downloading {filename}...")
    response = requests.get(url)
    with open(filepath, "wb") as f:
        f.write(response.content)

    print(f"Unzipping {filename}...")
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(filepath)
    print(f"Downloaded and extracted to: {os.path.abspath(extract_to)}")

def main():
    parser = argparse.ArgumentParser(description="Download and extract a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["african-wildlife"],
        help="Download various datasets.",
    )
    parser.add_argument(
        "--o",
        type=str,
        default=".",
        help="Directory to save the downloaded and extracted dataset.",
    )

    args = parser.parse_args()

    if args.dataset in dataset_urls:
        target_dir = os.path.join(args.o, args.dataset)
        os.makedirs(target_dir, exist_ok=True)
        download_and_unzip(dataset_urls[args.dataset], extract_to=target_dir)
    else:
        print("Dataset not found.")


if __name__ == "__main__":
    main()
    # Example usage: python download_dataset.py --dataset african-wildlife --o .