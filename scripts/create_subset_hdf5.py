#!/usr/bin/env python3
"""
Trích subset nhỏ từ file HDF5 lớn để test trên RTX 2050.

Usage:
    python scripts/create_subset_hdf5.py \
        --input data_30h.h5 \
        --output data_test.h5 \
        --n_samples 100
"""
import argparse
import random
import h5py
import numpy as np
from tqdm import tqdm


def create_subset(input_file, output_file, n_samples=100, seed=42):
    random.seed(seed)

    with h5py.File(input_file, 'r') as fin:
        # Đọc danh sách mixture_ids
        if 'mixture_ids' in fin:
            all_ids = list(fin['mixture_ids'][:])
            if isinstance(all_ids[0], bytes):
                all_ids = [x.decode('utf-8') for x in all_ids]
        else:
            all_ids = list(fin['mixtures'].keys())

        total = len(all_ids)
        n_samples = min(n_samples, total)
        selected = sorted(random.sample(range(total), n_samples))
        selected_ids = [all_ids[i] for i in selected]

        print(f"Input:   {input_file} ({total:,} samples)")
        print(f"Output:  {output_file} ({n_samples} samples)")

        with h5py.File(output_file, 'w') as fout:
            # Copy root attributes
            for key, val in fin.attrs.items():
                fout.attrs[key] = val
            fout.attrs['n_samples'] = n_samples
            fout.attrs['total_duration_hours'] = (
                n_samples * int(fin.attrs.get('target_length', 64000))
                / int(fin.attrs['sample_rate']) / 3600
            )

            # Tạo group
            mix_grp = fout.create_group('mixtures')

            for mid in tqdm(selected_ids, desc="Copying"):
                src = fin['mixtures'][mid]
                dst = mix_grp.create_group(mid)

                # Copy datasets
                for key in src.keys():
                    dst.create_dataset(
                        key, data=src[key][:],
                        dtype='float32', compression='gzip',
                        compression_opts=1
                    )

                # Copy attributes
                for key, val in src.attrs.items():
                    dst.attrs[key] = val

            # Lưu mixture_ids
            dt = h5py.special_dtype(vlen=str)
            fout.create_dataset(
                'mixture_ids',
                data=np.array(selected_ids, dtype=object),
                dtype=dt
            )

    import os
    size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"\nDone! {output_file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input HDF5 file")
    parser.add_argument("--output", default="data_test.h5", help="Output subset file")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_subset(args.input, args.output, args.n_samples, args.seed)
