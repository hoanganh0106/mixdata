"""
H5Dataset - PyTorch Dataset đọc trực tiếp từ file HDF5.

Thay thế hoàn toàn việc đọc 108,000 file WAV rời rạc.
Một file data_30h.h5 duy nhất → sequential I/O → nhanh gấp 10-50× random reads.

Tương thích trực tiếp với SPMamba training pipeline.

Cấu trúc HDF5 expected:
    /mixtures/{mixture_id}/
        ├── mix   (float32, shape=[64000])
        ├── s1    (float32, shape=[64000])
        ├── s2    (float32, shape=[64000])
        └── s3    (float32, shape=[64000])
    /mixture_ids  (string array cho fast indexing)
    Root attrs: sample_rate, n_src, n_samples, target_length

Usage:
    from h5_dataset import H5Dataset, H5DataModule

    # === Dùng trực tiếp ===
    dataset = H5Dataset("data_30h.h5", segment=4.0)
    mixture, sources, name = dataset[0]
    # mixture: Tensor [64000]
    # sources: Tensor [3, 64000]

    # === Dùng với SPMamba pipeline ===
    datamodule = H5DataModule(
        h5_path="data_30h.h5",
        batch_size=1,      # auto-detect nếu dùng AutoVRAMConfig
        segment=2.0,
    )
"""
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class H5Dataset(Dataset):
    """
    PyTorch Dataset đọc trực tiếp từ HDF5.

    Đặc điểm:
    - Lazy loading: chỉ đọc data khi __getitem__ được gọi
    - File handle mở lazy per-worker (thread-safe cho DataLoader)
    - Hỗ trợ random crop segment cho training
    - Zero-copy nếu dùng numpy memmap

    Parameters:
        h5_path (str): Đường dẫn tới file .h5
        segment (float): Độ dài segment (giây). None = full length (test mode)
        sample_rate (int): Sample rate (đọc từ file attrs nếu không set)
        n_src (int): Số speakers (đọc từ file attrs nếu không set)
        normalize_audio (bool): Chuẩn hóa audio về zero-mean unit-variance
        subset_indices (list): Chỉ dùng subset các indices này (cho train/val split)
    """

    def __init__(
        self,
        h5_path: str,
        segment: float = 4.0,
        sample_rate: int = None,
        n_src: int = None,
        normalize_audio: bool = False,
        subset_indices: list = None,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.normalize_audio = normalize_audio
        self.EPS = 1e-8

        # Mở file 1 lần để đọc metadata, rồi đóng
        # (file handle sẽ được mở lazy trong __getitem__)
        with h5py.File(h5_path, 'r') as f:
            # Đọc metadata từ root attributes
            self.sample_rate = sample_rate or int(f.attrs['sample_rate'])
            self.n_src = n_src or int(f.attrs['n_src'])
            self.target_length = int(f.attrs.get('target_length', 64000))
            total_samples = int(f.attrs['n_samples'])

            # Đọc danh sách mixture_ids cho fast indexing
            if 'mixture_ids' in f:
                self.mixture_ids = list(f['mixture_ids'][:])
                # Decode bytes to str nếu cần
                if isinstance(self.mixture_ids[0], bytes):
                    self.mixture_ids = [mid.decode('utf-8') for mid in self.mixture_ids]
            else:
                # Fallback: liệt kê keys (chậm hơn cho file lớn)
                self.mixture_ids = list(f['mixtures'].keys())

        # Segment length
        if segment is None:
            self.seg_len = None
        else:
            self.seg_len = int(segment * self.sample_rate)

        self.test = self.seg_len is None

        # Subset filtering
        if subset_indices is not None:
            self.mixture_ids = [self.mixture_ids[i] for i in subset_indices]

        self.length = len(self.mixture_ids)

        # File handle (mở lazy per-worker)
        self._h5file = None

        print(f"[H5Dataset] Loaded {self.length} samples from {h5_path}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Speakers: {self.n_src}")
        print(f"  Segment: {segment}s" if segment else "  Segment: full length (test)")

    def _get_h5file(self):
        """
        Lazy-open file handle.
        Mỗi DataLoader worker cần mở handle riêng (HDF5 không thread-safe).
        """
        if self._h5file is None:
            self._h5file = h5py.File(self.h5_path, 'r')
        return self._h5file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._get_h5file()
        mid = self.mixture_ids[idx]
        grp = f['mixtures'][mid]

        # Đọc full-length arrays
        mix_data = grp['mix']
        total_len = len(mix_data)

        # Random crop cho training, full length cho testing
        if self.test or self.seg_len is None or total_len <= self.seg_len:
            rand_start = 0
            stop = total_len
        else:
            rand_start = np.random.randint(0, total_len - self.seg_len)
            stop = rand_start + self.seg_len

        # Đọc data slice (HDF5 supports efficient slicing)
        mixture = torch.from_numpy(mix_data[rand_start:stop].astype(np.float32))

        source_arrays = []
        for i in range(1, self.n_src + 1):
            s = grp[f's{i}'][rand_start:stop].astype(np.float32)
            source_arrays.append(s)
        sources = torch.from_numpy(np.stack(source_arrays))

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

        return mixture, sources, mid

    def __del__(self):
        """Đóng file handle khi object bị garbage collected."""
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass

    def get_stats(self):
        """Trả về thống kê dataset."""
        return {
            'h5_path': self.h5_path,
            'n_samples': self.length,
            'sample_rate': self.sample_rate,
            'n_src': self.n_src,
            'segment': self.seg_len / self.sample_rate if self.seg_len else None,
            'total_duration_hours': self.length * self.target_length / self.sample_rate / 3600,
            'file_size_gb': os.path.getsize(self.h5_path) / (1024**3),
        }


class H5DataModule:
    """
    DataModule cho SPMamba, đọc từ file HDF5.
    Drop-in replacement cho Libri3MixDataModule.

    Tự động chia train/val/test từ 1 file HDF5 duy nhất.

    Parameters:
        h5_path (str): Đường dẫn tới file .h5
        val_ratio (float): Tỷ lệ validation (default: 0.1 = 10%)
        test_ratio (float): Tỷ lệ test (default: 0.05 = 5%)
        segment (float): Segment length cho training (seconds)
        batch_size (int): Batch size
        num_workers (int): DataLoader workers
        seed (int): Random seed cho split reproducibility
    """

    def __init__(
        self,
        h5_path: str,
        val_ratio: float = 0.1,
        test_ratio: float = 0.05,
        segment: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 1,
        num_workers: int = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        seed: int = 42,
        sample_rate: int = 16000,
        n_src: int = 3,
        # Unused params for config compatibility
        **kwargs,
    ):
        self.h5_path = h5_path
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and (num_workers > 0)
        self.seed = seed
        self.sample_rate = sample_rate
        self.n_src = n_src

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self):
        """Chia dataset thành train/val/test bằng deterministic split."""
        # Đọc tổng số samples
        with h5py.File(self.h5_path, 'r') as f:
            total = int(f.attrs['n_samples'])

        # Deterministic split
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(total)

        n_test = int(total * self.test_ratio)
        n_val = int(total * self.val_ratio)
        n_train = total - n_val - n_test

        train_idx = sorted(indices[:n_train].tolist())
        val_idx = sorted(indices[n_train:n_train + n_val].tolist())
        test_idx = sorted(indices[n_train + n_val:].tolist())

        print(f"[H5DataModule] Split: train={n_train}, val={n_val}, test={n_test}")

        self.data_train = H5Dataset(
            h5_path=self.h5_path,
            segment=self.segment,
            sample_rate=self.sample_rate,
            n_src=self.n_src,
            normalize_audio=self.normalize_audio,
            subset_indices=train_idx,
        )
        self.data_val = H5Dataset(
            h5_path=self.h5_path,
            segment=self.segment,
            sample_rate=self.sample_rate,
            n_src=self.n_src,
            normalize_audio=self.normalize_audio,
            subset_indices=val_idx,
        )
        self.data_test = H5Dataset(
            h5_path=self.h5_path,
            segment=None,  # Full-length cho testing
            sample_rate=self.sample_rate,
            n_src=self.n_src,
            normalize_audio=self.normalize_audio,
            subset_indices=test_idx,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,  # batch=1 cho variable-length test
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    @property
    def make_loader(self):
        """Compatible với SPMamba audio_train.py"""
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test
