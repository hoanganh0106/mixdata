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

  echo "Generating LibriMix WAV files"
  $python_path scripts/create_librimix_from_metadata.py --librispeech_dir $librispeech_dir \
    --wham_dir $wham_dir \
    --metadata_dir $metadata_dir \
    --librimix_outdir $librimix_outdir \
    --n_src $n_src \
    --freqs 16k \
    --modes min \
    --types mix_both
done

# ============================================================
# Merge train-100 and train-360 vào một thư mục train/ duy nhất
# SPMamba cần cấu trúc: train/mix/ train/s1/ train/s2/ train/s3/
# ============================================================
echo "Merging train-100 and train-360 into single train/ directory..."

data_base="$librimix_outdir/Libri3Mix/wav16k/min"
train_dir="$data_base/train"

# Tạo thư mục đích
for subdir in mix s1 s2 s3; do
  mkdir -p "$train_dir/$subdir"
done

# Copy (hardlink) files từ train-100 và train-360 vào train/
for src_dir in "$data_base/train-100" "$data_base/train-360"; do
  if [ -d "$src_dir" ]; then
    for subdir in mix s1 s2 s3; do
      if [ -d "$src_dir/$subdir" ]; then
        cp -ln "$src_dir/$subdir/"*.wav "$train_dir/$subdir/" 2>/dev/null || \
        cp "$src_dir/$subdir/"*.wav "$train_dir/$subdir/"
      fi
    done
  fi
done

# Đếm số file
total_files=$(find "$train_dir" -name "*.wav" | wc -l)
echo ""
echo "=========================================="
echo "Dataset generation complete!"
echo "Data location: $train_dir"
echo "Total WAV files: $total_files (expected: 108,000)"
echo "=========================================="

if [ "$total_files" -ne 108000 ]; then
  echo "WARNING: Expected 108,000 files but found $total_files!"
fi
