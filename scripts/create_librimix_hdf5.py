#!/usr/bin/env python3
"""
Tạo dataset Libri3Mix dưới dạng HDF5 thay vì 108,000 file WAV rời rạc.
Ghi trực tiếp vào một file data_30h.h5 duy nhất.

Cấu trúc HDF5:
    /mixtures/{mixture_id}/
        ├── mix   (float32, shape=[64000])
        ├── s1    (float32, shape=[64000])
        ├── s2    (float32, shape=[64000])
        └── s3    (float32, shape=[64000])

Root Attributes:
    sample_rate:  16000
    n_src:        3
    n_samples:    27000
    duration_per_sample: 4.0
    total_duration_hours: 30.0
    created_at:   "2026-..."

Usage:
    python scripts/create_librimix_hdf5.py \
        --librispeech_dir mixdata/storage/LibriSpeech \
        --wham_dir mixdata/storage/wham_noise \
        --metadata_dir mixdata/storage/metadata/Libri3Mix \
        --output_file mixdata/storage/data_30h.h5 \
        --n_src 3 \
        --freq 16k \
        --batch_write_size 500
"""
import os
import sys
import argparse
import gc
import multiprocessing
from functools import partial
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

# ============================================================
# Constants (giữ nguyên từ script gốc)
# ============================================================
EPS = 1e-10
RATE = 16000
TARGET_LENGTH = 64000  # 4 seconds at 16kHz


# ============================================================
# Audio processing functions (giữ nguyên logic từ script gốc)
# ============================================================

def get_list_from_csv(row, column, n_src):
    """Trích list từ CSV row."""
    python_list = []
    for i in range(n_src):
        current_column = column.split('_')
        current_column.insert(1, str(i + 1))
        current_column = '_'.join(current_column)
        python_list.append(row[current_column])
    return python_list


def extend_noise(noise, max_length):
    """Nối dài noise bằng hanning window crossfade."""
    noise_ex = noise
    window = np.hanning(RATE + 1)
    i_w = window[:len(window) // 2 + 1]
    d_w = window[len(window) // 2::-1]
    while len(noise_ex) < max_length:
        noise_ex = np.concatenate((
            noise_ex[:len(noise_ex) - len(d_w)],
            np.multiply(noise_ex[len(noise_ex) - len(d_w):], d_w)
            + np.multiply(noise[:len(i_w)], i_w),
            noise[len(i_w):]
        ))
    noise_ex = noise_ex[:max_length]
    return noise_ex


def read_sources(row, n_src, librispeech_dir, wham_dir):
    """Đọc source audio files từ CSV row."""
    mixture_id = row['mixture_ID']
    sources_path_list = get_list_from_csv(row, 'source_path', n_src)
    gain_list = get_list_from_csv(row, 'source_gain', n_src)
    sources_list = []
    max_length = 0

    for sources_path in sources_path_list:
        sources_path = os.path.join(librispeech_dir, sources_path)
        source, _ = sf.read(sources_path, dtype='float32')
        if max_length < len(source):
            max_length = len(source)
        sources_list.append(source)

    # Read noise
    noise_path = os.path.join(wham_dir, row['noise_path'])
    noise, _ = sf.read(noise_path, dtype='float32', stop=max_length)
    if len(noise.shape) > 1:
        noise = noise[:, 0]
    if len(noise) < max_length:
        noise = extend_noise(noise, max_length)

    sources_list.append(noise)
    gain_list.append(row['noise_gain'])

    return mixture_id, gain_list, sources_list


def loudness_normalize(sources_list, gain_list):
    """Normalize sources loudness."""
    return [source * gain for source, gain in zip(sources_list, gain_list)]


def resample_list(sources_list, freq):
    """Resample sources to target frequency."""
    if freq == RATE:
        return sources_list
    return [resample_poly(source, freq, RATE) for source in sources_list]


def fit_lengths(source_list):
    """Cắt/pad tất cả sources về TARGET_LENGTH (64000 samples = 4s)."""
    reshaped = []
    for source in source_list:
        if len(source) > TARGET_LENGTH:
            reshaped.append(source[:TARGET_LENGTH])
        else:
            reshaped.append(
                np.pad(source, (0, TARGET_LENGTH - len(source)), mode='constant')
            )
    return reshaped


def mix_sources(sources_list):
    """Mix tất cả sources thành mixture."""
    mixture = np.zeros_like(sources_list[0])
    for source in sources_list:
        mixture += source
    return mixture


def compute_snr(mixture, source):
    """Compute SNR giữa source và noise trong mixture."""
    noise = mixture - source
    return 10 * np.log10(np.mean(source ** 2) / (np.mean(noise ** 2) + EPS) + EPS)


def process_single_utterance(row, n_src, librispeech_dir, wham_dir, freq):
    """
    Xử lý 1 utterance: đọc sources → normalize → resample → fit → mix.
    Trả về dict chứa audio arrays + metadata.
    KHÔNG ghi file, chỉ trả về data trong RAM.
    """
    # Read sources
    mixture_id, gain_list, sources_list = read_sources(
        row, n_src, librispeech_dir, wham_dir
    )

    # Transform: normalize → resample → fit to 4s
    sources_norm = loudness_normalize(sources_list, gain_list)
    sources_resampled = resample_list(sources_norm, freq)
    sources_fitted = fit_lengths(sources_resampled)

    # Tạo mixture (mix_both = tất cả sources + noise)
    mixture = mix_sources(sources_fitted)

    # Compute SNR cho mỗi speaker
    snr_list = [compute_snr(mixture, sources_fitted[i]) for i in range(n_src)]

    return {
        'mixture_id': mixture_id,
        'mix': mixture.astype(np.float32),
        's1': sources_fitted[0].astype(np.float32),
        's2': sources_fitted[1].astype(np.float32),
        's3': sources_fitted[2].astype(np.float32),
        'snr': snr_list,
        'length': len(mixture),
    }


# ============================================================
# HDF5 Writer - Ghi theo batch để tránh tràn RAM
# ============================================================

class HDF5BatchWriter:
    """
    Ghi dữ liệu vào HDF5 theo batch.
    Tích lũy samples trong buffer, flush khi đầy.
    
    Tránh tràn RAM trên RTX 2050 (thường đi kèm 8-16GB RAM):
    - Mỗi sample: 4 arrays × 64000 × 4 bytes = ~1 MB
    - batch_size=500 → buffer ~500 MB
    """

    def __init__(self, output_file, n_src=3, sample_rate=16000,
                 batch_write_size=500):
        self.output_file = output_file
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.batch_write_size = batch_write_size

        # Buffer
        self.buffer = []
        self.total_written = 0

        # Tạo file HDF5 mới
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        self.h5file = h5py.File(output_file, 'w')

        # Tạo group chính
        self.mixtures_group = self.h5file.create_group('mixtures')

        # Tạo group để lưu danh sách mixture_ids (cho indexing nhanh)
        # Sẽ được ghi lại khi close()
        self.mixture_ids = []

        print(f"[HDF5] Opened {output_file} for writing")
        print(f"[HDF5] Batch write size: {batch_write_size} samples")

    def add_sample(self, sample_dict):
        """Thêm 1 sample vào buffer."""
        self.buffer.append(sample_dict)

        if len(self.buffer) >= self.batch_write_size:
            self.flush()

    def flush(self):
        """Ghi buffer xuống HDF5 rồi giải phóng RAM."""
        if not self.buffer:
            return

        for sample in self.buffer:
            mid = sample['mixture_id']
            grp = self.mixtures_group.create_group(mid)

            # Ghi 4 arrays audio (float32)
            # Dùng compression để giảm kích thước file
            grp.create_dataset('mix', data=sample['mix'],
                               dtype='float32', compression='gzip',
                               compression_opts=1)
            grp.create_dataset('s1', data=sample['s1'],
                               dtype='float32', compression='gzip',
                               compression_opts=1)
            grp.create_dataset('s2', data=sample['s2'],
                               dtype='float32', compression='gzip',
                               compression_opts=1)
            grp.create_dataset('s3', data=sample['s3'],
                               dtype='float32', compression='gzip',
                               compression_opts=1)

            # Lưu metadata vào attributes của group
            grp.attrs['length'] = sample['length']
            grp.attrs['duration'] = sample['length'] / self.sample_rate
            for i, snr in enumerate(sample['snr']):
                grp.attrs[f's{i+1}_snr'] = float(snr)

            self.mixture_ids.append(mid)

        self.total_written += len(self.buffer)
        n_flushed = len(self.buffer)

        # Giải phóng RAM
        self.buffer.clear()
        gc.collect()

        print(f"[HDF5] Flushed {n_flushed} samples "
              f"(total: {self.total_written})")

    def close(self):
        """Flush buffer còn lại, ghi metadata, đóng file."""
        self.flush()

        # Ghi root-level attributes
        self.h5file.attrs['sample_rate'] = self.sample_rate
        self.h5file.attrs['n_src'] = self.n_src
        self.h5file.attrs['n_samples'] = self.total_written
        self.h5file.attrs['duration_per_sample'] = TARGET_LENGTH / self.sample_rate
        self.h5file.attrs['total_duration_hours'] = (
            self.total_written * TARGET_LENGTH / self.sample_rate / 3600
        )
        self.h5file.attrs['target_length'] = TARGET_LENGTH
        self.h5file.attrs['created_at'] = datetime.now().isoformat()

        # Ghi danh sách mixture_ids để indexing nhanh
        # (tránh phải liệt kê keys mỗi lần load)
        dt = h5py.special_dtype(vlen=str)
        ids_dataset = self.h5file.create_dataset(
            'mixture_ids',
            data=np.array(self.mixture_ids, dtype=object),
            dtype=dt
        )

        self.h5file.close()

        # In thống kê
        file_size = os.path.getsize(self.output_file)
        print(f"\n{'='*60}")
        print(f"  HDF5 DATASET CREATED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"  File:       {self.output_file}")
        print(f"  Size:       {file_size / (1024**3):.2f} GB")
        print(f"  Samples:    {self.total_written:,}")
        print(f"  Duration:   {self.total_written * 4 / 3600:.1f} hours")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Speakers:   {self.n_src}")
        print(f"{'='*60}")


# ============================================================
# Main pipeline
# ============================================================

def _worker_wrapper(row_tuple, n_src, librispeech_dir, wham_dir, freq):
    idx, row = row_tuple
    try:
        sample = process_single_utterance(
            row, n_src, librispeech_dir, wham_dir, freq
        )
        return idx, sample, None
    except Exception as e:
        return idx, None, str(e)

def process_metadata_to_hdf5(csv_path, writer, n_src, librispeech_dir,
                              wham_dir, freq):
    """Xử lý 1 CSV metadata file, ghi kết quả vào HDF5 writer."""
    print(f"\nProcessing: {csv_path}")
    md_file = pd.read_csv(csv_path, engine='python')
    total_rows = len(md_file)
    print(f"  Rows: {total_rows}")

    # Giảm số processes xuống để tránh lỗi hết RAM (The paging file is too small)
    # Spawning 60 processes x ~200MB RAM each = 12GB RAM just for Python environments alone.
    num_procs = min(multiprocessing.cpu_count() - 2, 8) 
    print(f"  Sử dụng {num_procs} processes xử lý song song")
    worker_func = partial(
        _worker_wrapper,
        n_src=n_src, 
        librispeech_dir=librispeech_dir, 
        wham_dir=wham_dir, 
        freq=freq
    )

    with multiprocessing.Pool(processes=num_procs) as pool:
        for idx, sample, err in tqdm(pool.imap_unordered(worker_func, md_file.iterrows()),
                                     total=total_rows,
                                     desc="  Generating"):
            if err:
                print(f"\n  [WARN] Failed to process row {idx}: {err}")
            else:
                writer.add_sample(sample)

    print(f"  Done: {total_rows} utterances processed")


def main(args):
    librispeech_dir = args.librispeech_dir
    wham_dir = args.wham_dir
    metadata_dir = args.metadata_dir
    output_file = args.output_file
    n_src = args.n_src
    batch_write_size = args.batch_write_size

    # Parse frequency
    freq_str = args.freq.lower().strip('k')
    freq = int(freq_str) * 1000

    print("=" * 60)
    print("  LIBRI3MIX HDF5 GENERATOR")
    print("=" * 60)
    print(f"  LibriSpeech dir:  {librispeech_dir}")
    print(f"  WHAM dir:         {wham_dir}")
    print(f"  Metadata dir:     {metadata_dir}")
    print(f"  Output file:      {output_file}")
    print(f"  Speakers (n_src): {n_src}")
    print(f"  Frequency:        {freq} Hz")
    print(f"  Batch write size: {batch_write_size}")
    print(f"  Target length:    {TARGET_LENGTH} samples ({TARGET_LENGTH/freq:.1f}s)")
    print("=" * 60)

    # Kiểm tra metadata files
    md_files = sorted([
        f for f in os.listdir(metadata_dir)
        if f.endswith('.csv') and 'info' not in f
    ])

    if not md_files:
        print(f"[ERROR] No CSV metadata files found in {metadata_dir}")
        sys.exit(1)

    print(f"\nFound {len(md_files)} metadata files:")
    for f in md_files:
        df = pd.read_csv(os.path.join(metadata_dir, f), engine='python')
        print(f"  {f}: {len(df)} mixtures")

    # Tạo HDF5 writer
    writer = HDF5BatchWriter(
        output_file=output_file,
        n_src=n_src,
        sample_rate=freq,
        batch_write_size=batch_write_size,
    )

    # Xử lý từng metadata file
    try:
        for md_file in md_files:
            csv_path = os.path.join(metadata_dir, md_file)
            process_metadata_to_hdf5(
                csv_path, writer, n_src, librispeech_dir, wham_dir, freq
            )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving partial data...")
    finally:
        writer.close()

    print("\nDone! File ready for training.")
    print(f"Use with: H5Dataset('{output_file}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Libri3Mix dataset as a single HDF5 file"
    )
    parser.add_argument('--librispeech_dir', type=str, required=True,
                        help='Path to LibriSpeech root directory')
    parser.add_argument('--wham_dir', type=str, required=True,
                        help='Path to wham_noise root directory')
    parser.add_argument('--metadata_dir', type=str, required=True,
                        help='Path to LibriMix metadata directory (contains CSV files)')
    parser.add_argument('--output_file', type=str, default='data_30h.h5',
                        help='Output HDF5 file path (default: data_30h.h5)')
    parser.add_argument('--n_src', type=int, default=3,
                        help='Number of source speakers (default: 3)')
    parser.add_argument('--freq', type=str, default='16k',
                        help='Target frequency (default: 16k)')
    parser.add_argument('--batch_write_size', type=int, default=500,
                        help='Number of samples to buffer before flushing to disk. '
                             'Lower = less RAM usage. (default: 500, ~500MB RAM)')
    args = parser.parse_args()
    main(args)
