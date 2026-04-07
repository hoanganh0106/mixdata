#!/usr/bin/env python3
"""
generate_librimix_hdf5.py
Windows-native equivalent of generate_librimix_hdf5.sh
Tải LibriSpeech + WHAM, tạo metadata, ghi HDF5, dọn dẹp.
"""

import os
import sys
import shutil
import tarfile
import zipfile
import subprocess
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Config
# ============================================================
storage_dir = Path("mixdata/storage")
librispeech_dir = storage_dir / "LibriSpeech"
wham_dir = storage_dir / "wham_noise"
librimix_outdir = storage_dir

python_path = sys.executable  # Dùng chính Python hiện tại (trong venv)


# ============================================================
# Download helpers
# ============================================================

class DownloadProgressBar:
    """Simple progress indicator for urllib downloads."""
    def __init__(self, filename):
        self.filename = filename
        self.last_percent = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size > 0:
            percent = int(block_num * block_size * 100 / total_size)
            percent = min(percent, 100)
            if percent != self.last_percent:
                self.last_percent = percent
                bar = '█' * (percent // 2) + '░' * (50 - percent // 2)
                print(f"\r  [{bar}] {percent}% - {self.filename}", end='', flush=True)
                if percent == 100:
                    print()


def download_file(url, dest_path):
    """Download file if it doesn't exist, with resume support via wget-style check."""
    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"  File already downloaded: {dest_path}")
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, str(dest_path), DownloadProgressBar(dest_path.name))
    return dest_path


# ============================================================
# Download functions
# ============================================================

def download_librispeech_clean100():
    target = librispeech_dir / "train-clean-100"
    if target.exists():
        print("[OK] LibriSpeech/train-clean-100 already exists")
        return
    print("Download LibriSpeech/train-clean-100...")
    archive = download_file(
        "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
        storage_dir / "train-clean-100.tar.gz"
    )
    print("  Extracting train-clean-100.tar.gz ...")
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(path=str(storage_dir))
    archive.unlink()
    print("[OK] train-clean-100 extracted")


def download_librispeech_clean360():
    target = librispeech_dir / "train-clean-360"
    if target.exists():
        print("[OK] LibriSpeech/train-clean-360 already exists")
        return
    print("Download LibriSpeech/train-clean-360...")
    archive = download_file(
        "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
        storage_dir / "train-clean-360.tar.gz"
    )
    print("  Extracting train-clean-360.tar.gz ...")
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(path=str(storage_dir))
    archive.unlink()
    print("[OK] train-clean-360 extracted")


def download_wham():
    if wham_dir.exists():
        print("[OK] wham_noise already exists")
        return
    print("Download wham_noise...")
    archive = download_file(
        "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip",
        storage_dir / "wham_noise.zip"
    )
    print("  Extracting wham_noise.zip ...")
    with zipfile.ZipFile(archive, 'r') as zf:
        zf.extractall(path=str(storage_dir))
    archive.unlink()
    print("[OK] wham_noise extracted")


# ============================================================
# Run a python sub-script
# ============================================================

def run_script(script, *args):
    """Run a Python script as subprocess, streaming output."""
    cmd = [python_path, script] + list(args)
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


# ============================================================
# Main pipeline
# ============================================================

def main():
    print("=" * 50)
    print("  LibriMix HDF5 Pipeline (Windows native)")
    print("=" * 50)

    # --- Step 1: Download song song ---
    print("\n[Step 1/6] Downloading datasets...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(download_librispeech_clean100): "clean-100",
            pool.submit(download_librispeech_clean360): "clean-360",
            pool.submit(download_wham): "wham_noise",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Download {name} failed: {e}")
                sys.exit(1)

    # --- Step 2: Augment noise ---
    print("\n[Step 2/6] Augmenting train noise...")
    run_script("scripts/augment_train_noise.py",
               "--wham_dir", str(wham_dir))

    # --- Step 3: LibriSpeech metadata ---
    print("\n[Step 3/6] Generating LibriSpeech metadata...")
    run_script("scripts/create_librispeech_metadata.py",
               "--librispeech_dir", str(librispeech_dir),
               "--librispeech_md_dir", str(storage_dir / "metadata" / "LibriSpeech"))

    # --- Step 4: WHAM metadata ---
    print("\n[Step 4/6] Generating WHAM metadata...")
    run_script("scripts/create_wham_metadata.py",
               "--wham_dir", str(wham_dir),
               "--wham_md_dir", str(storage_dir / "metadata" / "Wham_noise"))

    # --- Step 5: LibriMix metadata + HDF5 ---
    n_src = 3
    metadata_dir = storage_dir / "metadata" / f"Libri{n_src}Mix"

    print(f"\n[Step 5/6] Generating Libri{n_src}Mix metadata...")
    run_script("scripts/create_librimix_metadata.py",
               "--librispeech_dir", str(librispeech_dir),
               "--librispeech_md_dir", str(storage_dir / "metadata" / "LibriSpeech"),
               "--wham_dir", str(wham_dir),
               "--wham_md_dir", str(storage_dir / "metadata" / "Wham_noise"),
               "--metadata_outdir", str(metadata_dir),
               "--n_src", str(n_src))

    hdf5_output = storage_dir / "data_30h.h5"
    print(f"\n[Step 6/6] Generating HDF5 dataset: {hdf5_output}")
    print("  (Thay thế 108,000 file WAV rời rạc)")
    run_script("scripts/create_librimix_hdf5.py",
               "--librispeech_dir", str(librispeech_dir),
               "--wham_dir", str(wham_dir),
               "--metadata_dir", str(metadata_dir),
               "--output_file", str(hdf5_output),
               "--n_src", str(n_src),
               "--freq", "16k",
               "--batch_write_size", "10000")

    # --- Cleanup ---
    print("\nCleaning up source files...")
    if librispeech_dir.exists():
        shutil.rmtree(librispeech_dir)
        print("  [OK] Removed LibriSpeech (~30 GB)")
    if wham_dir.exists():
        shutil.rmtree(wham_dir)
        print("  [OK] Removed WHAM noise (~5 GB)")
    md_dir = storage_dir / "metadata"
    if md_dir.exists():
        shutil.rmtree(md_dir)
        print("  [OK] Removed CSV metadata")

    # --- Done ---
    h5_size = hdf5_output.stat().st_size / (1024 ** 3)
    print()
    print("=" * 50)
    print("  DATASET GENERATION COMPLETE!")
    print()
    print(f"  HDF5 file: {hdf5_output}")
    print(f"  Size: {h5_size:.2f} GB")
    print(f"  Disk saved: ~35 GB (source files removed)")
    print()
    print(f"  Training:")
    print(f"    python audio_train_h5.py --h5_path {hdf5_output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
