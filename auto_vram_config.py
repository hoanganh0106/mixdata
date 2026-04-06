"""
Auto-VRAM Configuration cho SPMamba.

Tự động dò tìm GPU VRAM và điều chỉnh batch_size, segment, precision
để chạy được trên mọi loại GPU mà không cần sửa config thủ công.

Tích hợp trực tiếp với PyTorch Lightning.

Usage:
    from auto_vram_config import AutoVRAMConfig, create_auto_trainer

    # === Dùng đơn giản ===
    config = AutoVRAMConfig.detect()
    print(config)
    # → AutoVRAMConfig(gpu=RTX 2050, vram=4.0GB, batch_size=1, segment=2.0, ...)

    # === Tích hợp với SPMamba ===
    # Trong audio_train.py, thay thế trainer setup:
    trainer = create_auto_trainer(config, exp_dir, logger)

    # === Dùng với H5DataModule ===
    from h5_dataset import H5DataModule
    datamodule = H5DataModule(
        h5_path="data_30h.h5",
        batch_size=config.batch_size,
        segment=config.segment,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
"""
import torch
import os
import platform
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AutoVRAMConfig:
    """
    Cấu hình tự động dựa trên VRAM GPU.

    Tiers:
        VRAM < 6GB   → RTX 2050/3050/1650    → batch=1, seg=2.0s, fp16
        VRAM 6-12GB  → RTX 3060/3070/4060     → batch=4, seg=4.0s, fp16
        VRAM 12-24GB → RTX 3090/4090/A5000    → batch=8, seg=4.0s, bf16
        VRAM 24-48GB → A6000/L40              → batch=16, seg=4.0s, bf16
        VRAM > 48GB  → A100/H100             → batch=32, seg=4.0s, bf16
    """
    # GPU info
    gpu_name: str = "Unknown"
    vram_gb: float = 0.0
    vram_total_bytes: int = 0
    compute_capability: tuple = (0, 0)

    # Auto-detected config
    batch_size: int = 1
    segment: float = 2.0
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 1
    num_workers: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False
    strategy: str = "auto"
    num_gpus: int = 1

    # Effective batch size (batch × gpus × accumulate)
    effective_batch_size: int = 1

    # Tier name
    tier: str = "unknown"

    @classmethod
    def detect(cls) -> 'AutoVRAMConfig':
        """
        Tự động dò tìm GPU và trả về config phù hợp.
        Nếu không có CUDA, trả về CPU config.
        """
        config = cls()

        if not torch.cuda.is_available():
            print("[AutoVRAM] No CUDA GPU detected → CPU mode")
            config.tier = "cpu"
            config.gpu_name = "CPU"
            config.batch_size = 1
            config.segment = 1.0
            config.precision = "32"
            config.num_workers = 2
            return config

        # Đọc thông tin GPU
        props = torch.cuda.get_device_properties(0)
        config.gpu_name = props.name
        config.vram_total_bytes = props.total_mem
        config.vram_gb = props.total_mem / (1024 ** 3)
        config.compute_capability = (props.major, props.minor)
        config.num_gpus = torch.cuda.device_count()

        # Kiểm tra bf16 support (compute capability >= 8.0)
        supports_bf16 = props.major >= 8

        print(f"[AutoVRAM] Detected GPU: {config.gpu_name}")
        print(f"[AutoVRAM] VRAM: {config.vram_gb:.1f} GB")
        print(f"[AutoVRAM] Compute capability: {props.major}.{props.minor}")
        print(f"[AutoVRAM] BF16 support: {'Yes' if supports_bf16 else 'No'}")
        print(f"[AutoVRAM] GPU count: {config.num_gpus}")

        # === TIER DETECTION ===

        vram = config.vram_gb

        if vram < 6:
            # --- Tier 1: RTX 2050, GTX 1650, etc ---
            config.tier = "low_vram"
            config.batch_size = 1
            config.segment = 2.0       # Giảm segment để fit VRAM
            config.precision = "16-mixed"
            config.accumulate_grad_batches = 8  # Simulate batch=8
            config.num_workers = 2
            config.pin_memory = False
            config.persistent_workers = False
            config.strategy = "auto"   # Single GPU

        elif vram < 12:
            # --- Tier 2: RTX 3060, RTX 4060, etc ---
            config.tier = "mid_vram"
            config.batch_size = 4
            config.segment = 4.0
            config.precision = "16-mixed"
            config.accumulate_grad_batches = 4
            config.num_workers = 4
            config.pin_memory = True
            config.persistent_workers = True
            config.strategy = "auto"

        elif vram < 24:
            # --- Tier 3: RTX 3090, RTX 4090, A5000, etc ---
            config.tier = "high_vram"
            config.batch_size = 8
            config.segment = 4.0
            config.precision = "bf16-mixed" if supports_bf16 else "16-mixed"
            config.accumulate_grad_batches = 2
            config.num_workers = 16 # Tăng mạnh cực đại do có 112 Threads CPU
            config.pin_memory = True
            config.persistent_workers = True
            config.strategy = "auto"

        elif vram < 48:
            # --- Tier 4: A6000, L40, etc ---
            config.tier = "pro_vram"
            config.batch_size = 16
            config.segment = 4.0
            config.precision = "bf16-mixed" if supports_bf16 else "16-mixed"
            config.accumulate_grad_batches = 1
            config.num_workers = 8
            config.pin_memory = True
            config.persistent_workers = True
            config.strategy = "auto"

        else:
            # --- Tier 5: A100 (80GB), H100), multi-GPU ---
            config.tier = "ultra_vram"
            config.batch_size = 32
            config.segment = 4.0
            config.precision = "bf16-mixed"
            config.accumulate_grad_batches = 1
            config.num_workers = 8
            config.pin_memory = True
            config.persistent_workers = True

            if config.num_gpus > 1:
                config.strategy = "ddp"
            else:
                config.strategy = "auto"

        # Effective batch size
        config.effective_batch_size = (
            config.batch_size * config.num_gpus * config.accumulate_grad_batches
        )

        # In summary
        print(f"\n[AutoVRAM] === Auto Configuration ===")
        print(f"  Tier:             {config.tier}")
        print(f"  batch_size:       {config.batch_size}")
        print(f"  segment:          {config.segment}s")
        print(f"  precision:        {config.precision}")
        print(f"  accumulate_grad:  {config.accumulate_grad_batches}")
        print(f"  effective_batch:  {config.effective_batch_size}")
        print(f"  num_workers:      {config.num_workers}")
        print(f"  pin_memory:       {config.pin_memory}")
        print(f"  strategy:         {config.strategy}")
        print()

        return config

    def to_dict(self):
        """Convert to dict cho YAML/JSON serialization."""
        return {
            'gpu_name': self.gpu_name,
            'vram_gb': round(self.vram_gb, 1),
            'tier': self.tier,
            'batch_size': self.batch_size,
            'segment': self.segment,
            'precision': self.precision,
            'accumulate_grad_batches': self.accumulate_grad_batches,
            'effective_batch_size': self.effective_batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'strategy': self.strategy,
            'num_gpus': self.num_gpus,
        }


def create_auto_trainer(config: AutoVRAMConfig, exp_dir: str,
                         logger=None, max_epochs=100, callbacks=None):
    """
    Tạo PyTorch Lightning Trainer với auto-VRAM config.
    Drop-in replacement cho trainer setup trong audio_train.py.

    Parameters:
        config: AutoVRAMConfig từ AutoVRAMConfig.detect()
        exp_dir: Experiment directory cho checkpoints
        logger: Lightning logger (TensorBoard/WandB)
        max_epochs: Số epochs tối đa
        callbacks: List of Lightning callbacks

    Returns:
        pl.Trainer instance
    """
    import pytorch_lightning as pl

    # Strategy
    if config.strategy == "ddp" and config.num_gpus > 1:
        from pytorch_lightning.strategies.ddp import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"

    # Devices
    if torch.cuda.is_available():
        devices = config.num_gpus
        accelerator = "cuda"
    else:
        devices = 1
        accelerator = "cpu"

    trainer = pl.Trainer(
        precision=config.precision,
        max_epochs=max_epochs,
        callbacks=callbacks or [],
        default_root_dir=exp_dir,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        gradient_clip_val=5.0,
        accumulate_grad_batches=config.accumulate_grad_batches,
        logger=logger,
        sync_batchnorm=(config.num_gpus > 1),
    )

    return trainer


def create_h5_datamodule_auto(h5_path: str, config: AutoVRAMConfig = None,
                                val_ratio=0.1, test_ratio=0.05,
                                normalize_audio=False, seed=42):
    """
    Tạo H5DataModule với auto-VRAM configuration.
    Tiện lợi function kết hợp VRAM detection + DataModule creation.

    Usage:
        datamodule = create_h5_datamodule_auto("data_30h.h5")
        datamodule.setup()
        train_loader, val_loader, test_loader = datamodule.make_loader
    """
    from h5_dataset import H5DataModule

    if config is None:
        config = AutoVRAMConfig.detect()

    return H5DataModule(
        h5_path=h5_path,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        segment=config.segment,
        normalize_audio=normalize_audio,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        seed=seed,
    )


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  AUTO-VRAM GPU DETECTOR")
    print("=" * 60)

    config = AutoVRAMConfig.detect()

    print("\n" + "=" * 60)
    print("  FULL CONFIG")
    print("=" * 60)
    for k, v in config.to_dict().items():
        print(f"  {k:<25}: {v}")

    # Test nếu có file HDF5
    import sys
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
        print(f"\n[TEST] Creating DataModule from {h5_path}...")
        dm = create_h5_datamodule_auto(h5_path, config)
        dm.setup()

        train_loader, val_loader, test_loader = dm.make_loader
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches:   {len(val_loader)}")
        print(f"  Test batches:  {len(test_loader)}")

        # Test 1 batch
        for mix, src, name in train_loader:
            print(f"\n  Sample batch:")
            print(f"    mixture shape: {mix.shape}")
            print(f"    sources shape: {src.shape}")
            print(f"    name: {name}")
            break
