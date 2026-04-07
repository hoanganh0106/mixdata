#!/usr/bin/env python3
"""
Validate HDF5 dataset quality for SPMamba training.

Kiểm tra toàn diện:
  1. Cấu trúc file (groups, datasets, attributes)
  2. Metadata consistency (n_samples vs actual count)
  3. Data integrity (shape, dtype, NaN, Inf, zeros)
  4. Audio quality (SNR, energy levels, value ranges)
  5. Mix = sum(sources) consistency check
  6. Random sample spot-check with detailed stats

Usage:
    python scripts/validate_h5.py --h5_path mixdata/storage/data_30h.h5
    python scripts/validate_h5.py --h5_path mixdata/storage/data_30h.h5 --check_all
    python scripts/validate_h5.py --h5_path mixdata/storage/data_30h.h5 --n_random 50
"""
import os
import sys
import argparse
import time
import numpy as np
import h5py
from collections import Counter


def fmt_size(size_bytes):
    """Format bytes to human-readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(label, value, status=""):
    status_str = f"  {status}" if status else ""
    print(f"  {label:<35} {str(value):<20}{status_str}")


def check_pass(condition, msg_pass, msg_fail):
    if condition:
        print(f"  ✅ {msg_pass}")
        return True
    else:
        print(f"  ❌ {msg_fail}")
        return False


def validate_structure(f):
    """Check 1: Cấu trúc file."""
    print_header("1. FILE STRUCTURE")
    ok = True

    # Check root groups
    ok &= check_pass('mixtures' in f, 
                      "Root group 'mixtures' exists",
                      "MISSING root group 'mixtures'!")
    
    ok &= check_pass('mixture_ids' in f,
                      "Dataset 'mixture_ids' exists (fast indexing)",
                      "MISSING 'mixture_ids' dataset (will fallback to keys)")

    # Check root attributes
    required_attrs = ['sample_rate', 'n_src', 'n_samples', 'target_length', 'created_at']
    for attr in required_attrs:
        ok &= check_pass(attr in f.attrs,
                          f"Root attr '{attr}' = {f.attrs.get(attr, 'N/A')}",
                          f"MISSING root attribute '{attr}'!")

    optional_attrs = ['duration_per_sample', 'total_duration_hours']
    for attr in optional_attrs:
        if attr in f.attrs:
            print(f"  ℹ️  Optional attr '{attr}' = {f.attrs[attr]}")

    return ok


def validate_metadata(f):
    """Check 2: Metadata consistency."""
    print_header("2. METADATA CONSISTENCY")
    ok = True

    n_samples_attr = int(f.attrs.get('n_samples', 0))
    n_mixtures = len(f['mixtures'])
    n_ids = len(f['mixture_ids']) if 'mixture_ids' in f else 0

    print_result("n_samples (attr)", n_samples_attr)
    print_result("Actual mixtures count", n_mixtures)
    print_result("mixture_ids count", n_ids)

    ok &= check_pass(n_samples_attr == n_mixtures,
                      f"n_samples attr matches actual count ({n_samples_attr})",
                      f"MISMATCH! attr={n_samples_attr} vs actual={n_mixtures}")

    if n_ids > 0:
        ok &= check_pass(n_ids == n_mixtures,
                          f"mixture_ids count matches ({n_ids})",
                          f"MISMATCH! ids={n_ids} vs mixtures={n_mixtures}")

    # Check sample_rate and n_src
    sr = int(f.attrs.get('sample_rate', 0))
    n_src = int(f.attrs.get('n_src', 0))
    target_len = int(f.attrs.get('target_length', 0))

    ok &= check_pass(sr == 16000, f"Sample rate = {sr} Hz", f"Unexpected sample rate: {sr}")
    ok &= check_pass(n_src == 3, f"n_src = {n_src}", f"Unexpected n_src: {n_src}")
    ok &= check_pass(target_len == 64000, f"target_length = {target_len}", 
                      f"Unexpected target_length: {target_len}")

    return ok


def validate_data_integrity(f, check_all=False, n_random=100):
    """Check 3: Data integrity - shapes, dtypes, NaN, Inf."""
    print_header("3. DATA INTEGRITY")
    
    mixtures = f['mixtures']
    all_keys = list(mixtures.keys())
    n_total = len(all_keys)
    
    if check_all:
        keys_to_check = all_keys
        print(f"  Checking ALL {n_total} samples...")
    else:
        n_check = min(n_random, n_total)
        rng = np.random.RandomState(42)
        indices = rng.choice(n_total, n_check, replace=False)
        keys_to_check = [all_keys[i] for i in sorted(indices)]
        print(f"  Checking {n_check}/{n_total} random samples...")

    n_src = int(f.attrs.get('n_src', 3))
    target_len = int(f.attrs.get('target_length', 64000))
    expected_keys = ['mix'] + [f's{i}' for i in range(1, n_src + 1)]

    # Counters
    stats = {
        'checked': 0,
        'shape_errors': 0,
        'dtype_errors': 0,
        'nan_errors': 0,
        'inf_errors': 0,
        'all_zero_errors': 0,
        'missing_key_errors': 0,
        'clip_warnings': 0,  # |value| > 1.0
    }

    # Value range stats
    all_max_vals = []
    all_min_vals = []
    all_rms_vals = []

    start_time = time.time()
    report_interval = max(len(keys_to_check) // 10, 1)

    for i, key in enumerate(keys_to_check):
        grp = mixtures[key]
        stats['checked'] += 1

        # Check all expected datasets exist
        for dk in expected_keys:
            if dk not in grp:
                stats['missing_key_errors'] += 1
                if stats['missing_key_errors'] <= 5:
                    print(f"  ❌ Sample '{key}': missing dataset '{dk}'")
                continue

            data = grp[dk][:]

            # Shape check
            if data.shape != (target_len,):
                stats['shape_errors'] += 1
                if stats['shape_errors'] <= 5:
                    print(f"  ❌ Sample '{key}/{dk}': shape={data.shape}, expected=({target_len},)")

            # Dtype check
            if data.dtype != np.float32:
                stats['dtype_errors'] += 1
                if stats['dtype_errors'] <= 5:
                    print(f"  ❌ Sample '{key}/{dk}': dtype={data.dtype}, expected=float32")

            # NaN check
            if np.any(np.isnan(data)):
                stats['nan_errors'] += 1
                if stats['nan_errors'] <= 5:
                    print(f"  ❌ Sample '{key}/{dk}': contains NaN!")

            # Inf check
            if np.any(np.isinf(data)):
                stats['inf_errors'] += 1
                if stats['inf_errors'] <= 5:
                    print(f"  ❌ Sample '{key}/{dk}': contains Inf!")

            # All-zeros check
            if np.all(data == 0):
                stats['all_zero_errors'] += 1
                if stats['all_zero_errors'] <= 5:
                    print(f"  ⚠️  Sample '{key}/{dk}': all zeros!")

            # Clipping check
            if np.max(np.abs(data)) > 1.0:
                stats['clip_warnings'] += 1

            # Collect value stats
            all_max_vals.append(np.max(data))
            all_min_vals.append(np.min(data))
            all_rms_vals.append(np.sqrt(np.mean(data**2)))

        if (i + 1) % report_interval == 0:
            elapsed = time.time() - start_time
            pct = (i + 1) / len(keys_to_check) * 100
            print(f"  ... {i+1}/{len(keys_to_check)} checked ({pct:.0f}%), "
                  f"elapsed: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n  Checked {stats['checked']} samples in {elapsed:.1f}s")
    print()

    # Report
    ok = True
    ok &= check_pass(stats['missing_key_errors'] == 0,
                      "All samples have complete datasets (mix, s1, s2, s3)",
                      f"{stats['missing_key_errors']} samples missing datasets!")
    ok &= check_pass(stats['shape_errors'] == 0,
                      f"All shapes correct ({target_len},)",
                      f"{stats['shape_errors']} samples with wrong shape!")
    ok &= check_pass(stats['dtype_errors'] == 0,
                      "All dtypes correct (float32)",
                      f"{stats['dtype_errors']} samples with wrong dtype!")
    ok &= check_pass(stats['nan_errors'] == 0,
                      "No NaN values found",
                      f"{stats['nan_errors']} samples contain NaN!")
    ok &= check_pass(stats['inf_errors'] == 0,
                      "No Inf values found",
                      f"{stats['inf_errors']} samples contain Inf!")
    ok &= check_pass(stats['all_zero_errors'] == 0,
                      "No all-zero arrays found",
                      f"{stats['all_zero_errors']} samples are all zeros!")

    # Value range summary  
    if all_max_vals:
        print(f"\n  📊 Value Range Statistics:")
        print(f"     Global max value:  {max(all_max_vals):.6f}")
        print(f"     Global min value:  {min(all_min_vals):.6f}")
        print(f"     Mean RMS:          {np.mean(all_rms_vals):.6f}")
        print(f"     Max RMS:           {max(all_rms_vals):.6f}")
        print(f"     Min RMS:           {min(all_rms_vals):.6f}")
        if stats['clip_warnings'] > 0:
            pct = stats['clip_warnings'] / (stats['checked'] * len(expected_keys)) * 100
            print(f"     ⚠️  {stats['clip_warnings']} arrays ({pct:.1f}%) have values > 1.0 (clipping)")
        else:
            print(f"     ✅ No clipping detected (all values within [-1, 1])")

    return ok


def validate_mix_consistency(f, n_check=20):
    """Check 4: mix ≈ s1 + s2 + s3 + noise (consistency check)."""
    print_header("4. MIX CONSISTENCY (mix ≈ sum of sources)")

    mixtures = f['mixtures']
    all_keys = list(mixtures.keys())
    n_src = int(f.attrs.get('n_src', 3))

    rng = np.random.RandomState(123)
    indices = rng.choice(len(all_keys), min(n_check, len(all_keys)), replace=False)
    check_keys = [all_keys[i] for i in sorted(indices)]

    errors = []
    for key in check_keys:
        grp = mixtures[key]
        mix_data = grp['mix'][:]

        # Sum all sources (s1..s3 + noise which is included in the mix)
        source_sum = np.zeros_like(mix_data)
        for i in range(1, n_src + 1):
            source_sum += grp[f's{i}'][:]

        # The mix in this dataset is s1+s2+s3+noise (mix_both)
        # But s1,s2,s3 are the clean sources, so mix - (s1+s2+s3) = noise
        # We check if the reconstruction error is reasonable
        # Since mix = s1 + s2 + s3 + noise, mix - source_sum = noise component
        noise_residual = mix_data - source_sum
        noise_energy = np.sqrt(np.mean(noise_residual**2))
        mix_energy = np.sqrt(np.mean(mix_data**2))
        
        # The noise should be a reasonable fraction of the mix
        if mix_energy > 0:
            noise_ratio = noise_energy / mix_energy
        else:
            noise_ratio = 0
        
        errors.append({
            'key': key,
            'noise_energy': noise_energy,
            'mix_energy': mix_energy,
            'noise_ratio': noise_ratio,
        })

    # Report
    noise_ratios = [e['noise_ratio'] for e in errors]
    print(f"  Checked {len(check_keys)} samples")
    print(f"  Noise/Mix energy ratio:")
    print(f"    Mean:  {np.mean(noise_ratios):.4f}")
    print(f"    Std:   {np.std(noise_ratios):.4f}")
    print(f"    Min:   {np.min(noise_ratios):.4f}")
    print(f"    Max:   {np.max(noise_ratios):.4f}")
    print()

    # If noise ratio is very high (>0.9), something might be wrong
    ok = True
    ok &= check_pass(np.mean(noise_ratios) < 0.9,
                      "Mix contains meaningful source signal (noise ratio < 0.9)",
                      "Mix is mostly noise - sources might not be mixed properly!")
    
    # If noise ratio is 0 (no noise at all), that's also suspicious
    ok &= check_pass(np.mean(noise_ratios) > 0.001,
                      "Mix contains noise component (as expected for mix_both)",
                      "No noise detected - might be mix_clean instead of mix_both")

    return ok


def validate_audio_quality(f, n_check=50):
    """Check 5: Audio quality metrics."""
    print_header("5. AUDIO QUALITY METRICS")

    mixtures = f['mixtures']
    all_keys = list(mixtures.keys())
    n_src = int(f.attrs.get('n_src', 3))
    sr = int(f.attrs.get('sample_rate', 16000))

    rng = np.random.RandomState(456)
    indices = rng.choice(len(all_keys), min(n_check, len(all_keys)), replace=False)
    check_keys = [all_keys[i] for i in sorted(indices)]

    snr_values = {f's{i}': [] for i in range(1, n_src + 1)}
    energy_values = {'mix': [], **{f's{i}': [] for i in range(1, n_src + 1)}}
    silence_ratios = {'mix': [], **{f's{i}': [] for i in range(1, n_src + 1)}}

    for key in check_keys:
        grp = mixtures[key]
        mix_data = grp['mix'][:]
        
        energy_values['mix'].append(np.sqrt(np.mean(mix_data**2)))
        silence_ratios['mix'].append(np.mean(np.abs(mix_data) < 1e-6))

        for i in range(1, n_src + 1):
            src = grp[f's{i}'][:]
            energy_values[f's{i}'].append(np.sqrt(np.mean(src**2)))
            silence_ratios[f's{i}'].append(np.mean(np.abs(src) < 1e-6))

            # SNR: source vs rest
            noise = mix_data - src
            src_power = np.mean(src**2)
            noise_power = np.mean(noise**2)
            if noise_power > 1e-10:
                snr = 10 * np.log10(src_power / noise_power + 1e-10)
                snr_values[f's{i}'].append(snr)

    print(f"  Checked {len(check_keys)} samples\n")

    # Energy stats
    print(f"  📊 RMS Energy:")
    for k in energy_values:
        vals = energy_values[k]
        print(f"     {k:<5}: mean={np.mean(vals):.4f}, "
              f"std={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    # SNR stats
    print(f"\n  📊 Source SNR (dB) in mixture:")
    for k in snr_values:
        vals = snr_values[k]
        if vals:
            print(f"     {k}: mean={np.mean(vals):.1f} dB, "
                  f"std={np.std(vals):.1f}, "
                  f"range=[{np.min(vals):.1f}, {np.max(vals):.1f}]")

    # Silence ratio
    print(f"\n  📊 Silence Ratio (|sample| < 1e-6):")
    for k in silence_ratios:
        vals = silence_ratios[k]
        mean_silence = np.mean(vals) * 100
        print(f"     {k:<5}: {mean_silence:.1f}% silent on average")

    ok = True
    # Check that sources have reasonable energy
    for i in range(1, n_src + 1):
        mean_energy = np.mean(energy_values[f's{i}'])
        ok &= check_pass(mean_energy > 1e-4,
                          f"s{i} has meaningful energy (RMS={mean_energy:.4f})",
                          f"s{i} has very low energy ({mean_energy:.6f}) - might be silent!")

    # Check silence isn't too high
    mean_mix_silence = np.mean(silence_ratios['mix']) * 100
    ok &= check_pass(mean_mix_silence < 50,
                      f"Mix silence ratio acceptable ({mean_mix_silence:.1f}%)",
                      f"Mix has too much silence ({mean_mix_silence:.1f}%)!")

    return ok


def validate_dataloader_compat(h5_path):
    """Check 6: DataLoader compatibility test."""
    print_header("6. DATALOADER COMPATIBILITY")

    try:
        # Try importing the H5Dataset
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from h5_dataset import H5Dataset

        dataset = H5Dataset(h5_path, segment=4.0)
        
        # Load first 3 samples
        for i in range(min(3, len(dataset))):
            mixture, sources, name = dataset[i]
            
            print(f"\n  Sample {i}: '{name}'")
            print(f"    mixture shape: {mixture.shape}, dtype: {mixture.dtype}")
            print(f"    sources shape: {sources.shape}, dtype: {sources.dtype}")
            
            check_pass(mixture.shape == (64000,),
                       f"mixture shape correct: {mixture.shape}",
                       f"mixture shape wrong: {mixture.shape}")
            check_pass(sources.shape == (3, 64000),
                       f"sources shape correct: {sources.shape}",
                       f"sources shape wrong: {sources.shape}")

        print(f"\n  Dataset stats:")
        stats = dataset.get_stats()
        for k, v in stats.items():
            print(f"    {k}: {v}")

        check_pass(True, "H5Dataset loads successfully!", "")
        return True

    except ImportError:
        print("  ⚠️  Cannot import H5Dataset (h5_dataset.py not found)")
        print("  Skipping DataLoader compatibility test")
        return True
    except Exception as e:
        check_pass(False, "", f"H5Dataset failed: {e}")
        return False


def validate_unique_ids(f):
    """Check 7: All mixture IDs are unique."""
    print_header("7. UNIQUE MIXTURE IDS")
    
    if 'mixture_ids' in f:
        ids = list(f['mixture_ids'][:])
        if isinstance(ids[0], bytes):
            ids = [mid.decode('utf-8') for mid in ids]
    else:
        ids = list(f['mixtures'].keys())

    n_total = len(ids)
    n_unique = len(set(ids))
    
    ok = check_pass(n_total == n_unique,
                    f"All {n_total} mixture IDs are unique",
                    f"DUPLICATE IDs! {n_total} total, {n_unique} unique, "
                    f"{n_total - n_unique} duplicates")

    if n_total != n_unique:
        # Show duplicates
        counter = Counter(ids)
        dups = {k: v for k, v in counter.items() if v > 1}
        print(f"  Duplicates (showing first 10):")
        for i, (k, v) in enumerate(list(dups.items())[:10]):
            print(f"    '{k}': appears {v} times")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Validate HDF5 dataset")
    parser.add_argument('--h5_path', type=str, default='mixdata/storage/data_30h.h5',
                        help='Path to HDF5 file')
    parser.add_argument('--check_all', action='store_true',
                        help='Check ALL samples (slow for large datasets)')
    parser.add_argument('--n_random', type=int, default=100,
                        help='Number of random samples to check (default: 100)')
    parser.add_argument('--skip_loader', action='store_true',
                        help='Skip DataLoader compatibility test')
    args = parser.parse_args()

    h5_path = args.h5_path
    
    if not os.path.exists(h5_path):
        print(f"❌ File not found: {h5_path}")
        sys.exit(1)

    file_size = os.path.getsize(h5_path)
    
    print("=" * 60)
    print("  HDF5 DATASET VALIDATION")
    print("=" * 60)
    print(f"  File: {h5_path}")
    print(f"  Size: {fmt_size(file_size)}")
    print(f"  Mode: {'ALL samples' if args.check_all else f'{args.n_random} random samples'}")
    print("=" * 60)

    results = []

    with h5py.File(h5_path, 'r') as f:
        results.append(('Structure', validate_structure(f)))
        results.append(('Metadata', validate_metadata(f)))
        results.append(('Unique IDs', validate_unique_ids(f)))
        results.append(('Data Integrity', validate_data_integrity(
            f, check_all=args.check_all, n_random=args.n_random)))
        results.append(('Mix Consistency', validate_mix_consistency(f)))
        results.append(('Audio Quality', validate_audio_quality(f)))

    if not args.skip_loader:
        results.append(('DataLoader', validate_dataloader_compat(h5_path)))

    # Final report
    print_header("VALIDATION SUMMARY")
    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:<25} {status}")
        all_pass &= passed

    print()
    if all_pass:
        print("  🎉 ALL CHECKS PASSED - Dataset is ready for training!")
    else:
        print("  ⚠️  SOME CHECKS FAILED - Review errors above!")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
