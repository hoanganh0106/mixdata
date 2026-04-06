#!/bin/bash
set -eu  # Exit on error

storage_dir="mixdata/storage/"
librispeech_dir=$storage_dir/LibriSpeech
wham_dir=$storage_dir/wham_noise
librimix_outdir=$storage_dir/

# ============================================================
# Download functions — chỉ tải train-clean-100, train-clean-360
# và wham_noise (bỏ dev-clean, test-clean vì SPMamba tự chia)
# ============================================================

function LibriSpeech_clean100() {
	if ! test -e $librispeech_dir/train-clean-100; then
		echo "Download LibriSpeech/train-clean-100 into $storage_dir"
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-100.tar.gz -P $storage_dir
		tar -xzf $storage_dir/train-clean-100.tar.gz -C $storage_dir
		rm -rf $storage_dir/train-clean-100.tar.gz
	fi
}

function LibriSpeech_clean360() {
	if ! test -e $librispeech_dir/train-clean-360; then
		echo "Download LibriSpeech/train-clean-360 into $storage_dir"
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-360.tar.gz -P $storage_dir
		tar -xzf $storage_dir/train-clean-360.tar.gz -C $storage_dir
		rm -rf $storage_dir/train-clean-360.tar.gz
	fi
}

function wham() {
	if ! test -e $wham_dir; then
		echo "Download wham_noise into $storage_dir"
		wget -c --tries=0 --read-timeout=20 https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip -P $storage_dir
		unzip -qn $storage_dir/wham_noise.zip -d $storage_dir
		rm -rf $storage_dir/wham_noise.zip
	fi
}

# Download song song
LibriSpeech_clean100 &
LibriSpeech_clean360 &
wham &

wait

# Path to python
python_path=python3

# Augment noise (chỉ cần chạy 1 lần)
$python_path scripts/augment_train_noise.py --wham_dir $wham_dir

echo "Generating LibriSpeech metadata"
$python_path scripts/create_librispeech_metadata.py \
    --librispeech_dir $librispeech_dir \
    --librispeech_md_dir $storage_dir/metadata/LibriSpeech

echo "Generating WHAM metadata"
$python_path scripts/create_wham_metadata.py \
    --wham_dir $wham_dir \
    --wham_md_dir $storage_dir/metadata/Wham_noise

for n_src in 3; do
  metadata_dir=$storage_dir/metadata/Libri${n_src}Mix

  echo "Generating LibriMix metadata"
  $python_path scripts/create_librimix_metadata.py \
      --librispeech_dir $librispeech_dir \
      --librispeech_md_dir $storage_dir/metadata/LibriSpeech \
      --wham_dir $wham_dir \
      --wham_md_dir $storage_dir/metadata/Wham_noise \
      --metadata_outdir $metadata_dir \
      --n_src $n_src
done

# ============================================================
# GHI TRỰC TIẾP VÀO HDF5 THAY VÌ 108,000 FILE WAV
# Một file data_30h.h5 duy nhất (~8-10 GB compressed)
# ============================================================

hdf5_output="$storage_dir/data_30h.h5"
metadata_dir="$storage_dir/metadata/Libri3Mix"

echo ""
echo "============================================"
echo "  Generating HDF5 dataset: $hdf5_output"
echo "  (Thay thế 108,000 file WAV rời rạc)"
echo "============================================"

$python_path scripts/create_librimix_hdf5.py \
    --librispeech_dir $librispeech_dir \
    --wham_dir $wham_dir \
    --metadata_dir $metadata_dir \
    --output_file $hdf5_output \
    --n_src 3 \
    --freq 16k \
    --batch_write_size 500

# ============================================================
# DỌN DẸP: Xóa source audio + metadata (đã nằm trong HDF5)
# Tiết kiệm ~35 GB disk
# ============================================================

echo ""
echo "Cleaning up source files..."
rm -rf "$librispeech_dir"
echo "  [OK] Removed LibriSpeech (~30 GB)"
rm -rf "$wham_dir"
echo "  [OK] Removed WHAM noise (~5 GB)"
rm -rf "$storage_dir/metadata"
echo "  [OK] Removed CSV metadata"

echo ""
echo "============================================"
echo "  DATASET GENERATION COMPLETE!"
echo ""
echo "  HDF5 file: $hdf5_output"
echo "  Size: $(du -h $hdf5_output | cut -f1)"
echo "  Disk saved: ~35 GB (source files removed)"
echo ""
echo "  Training:"
echo "    python audio_train_h5.py --h5_path $hdf5_output"
echo "============================================"
