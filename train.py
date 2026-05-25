import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import dari file proyek
from model_dme import DensityMapRegressor
from dataset_loader import (
    DMEDataset, load_all_samples, stratified_split,
    get_train_transforms, get_val_transforms,
)


# ============================================================
# Hyperparameters
# ============================================================
EPOCHS = 30
BATCH_SIZE = 2          # Kecil karena komputasi heatmap cukup berat
LEARNING_RATE = 5e-5    # Learning rate untuk fine-tuning
CHECKPOINT_DIR = 'checkpoints'
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_dme_model.pth')


def select_device():
    """
    Deteksi device secara otomatis.
    Prioritas: CUDA (Nvidia GPU) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device   : CUDA — {gpu_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"  Device   : MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"  Device   : CPU")
    return device


def train():
    """
    Fungsi utama untuk training model DME.
    Training dari scratch (ImageNet pretrained weights) dengan stratified
    train/val split dan scale augmentation.
    """
    print("\n" + "=" * 65)
    print("  TRAINING — Density Map Estimation (DME)")
    print("=" * 65)

    # ---- 1. Device Selection ----
    device = select_device()

    # ---- 2. Buat folder checkpoint ----
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---- 3. Dataset & DataLoader ----
    print(f"\n  [Dataset]")

    # Load semua samples dan split secara stratified
    all_samples = load_all_samples()
    if len(all_samples) == 0:
        print("\n  [ERROR] Tidak ada data training!")
        print("  Pastikan sudah menjalankan:")
        print("    1. py point_labeler.py          (anotasi titik)")
        print("    2. Pastikan file JSON anotasi ada di dataset/annotations/")
        return

    train_samples, val_samples = stratified_split(all_samples, val_ratio=0.2)

    # Buat dataset objects
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    train_dataset = DMEDataset(
        samples=train_samples,
        transform=train_transform,
        scale_range=(0.75, 1.25),
    )
    val_dataset = DMEDataset(
        samples=val_samples,
        transform=val_transform,
        scale_range=(1.0, 1.0),  # Tidak ada scale augmentation saat validasi
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,      # 0 untuk Windows compatibility
        pin_memory=True if device.type == 'cuda' else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )

    print(f"  Train samples  : {len(train_dataset)}")
    print(f"  Val samples    : {len(val_dataset)}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Train batches  : {len(train_loader)}")
    print(f"  Val batches    : {len(val_loader)}")

    # ---- 4. Model (from scratch with ImageNet pretrained backbone) ----
    print(f"\n  [Model]")
    model = DensityMapRegressor(pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture   : MobileNetV2 + Dilated Conv + Upsample")
    print(f"  Total params   : {total_params:,}")
    print(f"  Trainable      : {trainable_params:,}")
    print(f"  Training from  : scratch (ImageNet pretrained backbone)")

    # ---- 5. Loss Function & Optimizer ----
    # MSELoss karena ini adalah regresi heatmap (bukan klasifikasi)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n  [Training Config]")
    print(f"  Loss function  : MSELoss")
    print(f"  Optimizer      : Adam")
    print(f"  Learning rate  : {LEARNING_RATE}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Checkpoint dir : {os.path.abspath(CHECKPOINT_DIR)}")

    # ---- 6. Training Loop ----
    best_val_mae = float('inf')

    print("\n" + "-" * 80)
    print(f"  {'Epoch':>5}  |  {'Train Loss':>12}  |  {'Train MAE':>10}  |  "
          f"{'Val MAE':>10}  |  {'Best Val MAE':>12}  |  Time")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ======== TRAINING PHASE ========
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        num_train_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            # Ambil data dari batch
            images = batch['image'].to(device)          # (B, 3, H, W)
            heatmaps = batch['heatmap'].to(device)      # (B, H, W)

            # Tambahkan dimensi channel pada heatmap: (B, H, W) -> (B, 1, H, W)
            heatmaps = heatmaps.unsqueeze(1)

            # ---- Forward pass ----
            outputs = model(images)                     # (B, 1, H, W)

            # ---- Hitung loss (MSE antara predicted & ground truth heatmap) ----
            loss = criterion(outputs, heatmaps * 1000.0)

            # ---- Backward pass & optimizer step ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- Hitung MAE (selisih jumlah objek) ----
            with torch.no_grad():
                batch_size_actual = images.size(0)
                for i in range(batch_size_actual):
                    pred_count = outputs[i].sum().item() / 1000.0
                    gt_count = heatmaps[i].sum().item()
                    epoch_mae += abs(pred_count - gt_count)

            epoch_loss += loss.item() * images.size(0)
            num_train_samples += images.size(0)

        avg_train_loss = epoch_loss / num_train_samples
        avg_train_mae = epoch_mae / num_train_samples

        # ======== VALIDATION PHASE ========
        model.eval()
        val_mae = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                heatmaps = batch['heatmap'].to(device)
                heatmaps = heatmaps.unsqueeze(1)

                outputs = model(images)

                for i in range(images.size(0)):
                    pred_count = outputs[i].sum().item() / 1000.0
                    gt_count = heatmaps[i].sum().item()  # NOT scaled by 1000
                    val_mae += abs(pred_count - gt_count)

        avg_val_mae = val_mae / len(val_dataset)

        epoch_time = time.time() - epoch_start

        # ---- Model Checkpointing (based on val MAE) ----
        is_best = avg_val_mae < best_val_mae
        if is_best:
            best_val_mae = avg_val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'train_mae': avg_train_mae,
                'loss': avg_train_loss,
            }, BEST_MODEL_PATH)
            marker = " * SAVED"
        else:
            marker = ""

        # ---- Logging ----
        print(f"  {epoch:>5}  |  {avg_train_loss:>12.8f}  |  {avg_train_mae:>10.4f}  |  "
              f"{avg_val_mae:>10.4f}  |  {best_val_mae:>12.4f}  |  "
              f"{epoch_time:.1f}s{marker}")

    # ---- Training Selesai ----
    print("-" * 80)
    print(f"\n  Training selesai!")
    print(f"  Best Val MAE   : {best_val_mae:.4f}")
    print(f"  Best model     : {os.path.abspath(BEST_MODEL_PATH)}")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    train()
