import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import dari file proyek
from model_dme import DensityMapRegressor
from dataset_loader import DMEDataset, get_train_transforms


# ============================================================
# Hyperparameters
# ============================================================
EPOCHS = 50
BATCH_SIZE = 2          # Kecil karena komputasi heatmap cukup berat
LEARNING_RATE = 1e-4
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
    train_transform = get_train_transforms()
    train_dataset = DMEDataset(
        images_dir=os.path.join('dataset', 'images'),
        ground_truth_dir=os.path.join('dataset', 'ground_truth'),
        transform=train_transform,
    )

    if len(train_dataset) == 0:
        print("\n  [ERROR] Tidak ada data training!")
        print("  Pastikan sudah menjalankan:")
        print("    1. py point_labeler.py          (anotasi titik)")
        print("    2. py generate_ground_truth.py   (generate .npy)")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,      # 0 untuk Windows compatibility
        pin_memory=True if device.type == 'cuda' else False,
    )

    print(f"  Total samples  : {len(train_dataset)}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Total batches  : {len(train_loader)}")

    # ---- 4. Model ----
    print(f"\n  [Model]")
    model = DensityMapRegressor(pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture   : MobileNetV2 + Dilated Conv + Upsample")
    print(f"  Total params   : {total_params:,}")
    print(f"  Trainable      : {trainable_params:,}")

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
    best_mae = float('inf')

    print("\n" + "-" * 65)
    print(f"  {'Epoch':>5}  |  {'Loss':>12}  |  {'MAE':>10}  |  {'Best MAE':>10}  |  Time")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        num_samples = 0

        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Ambil data dari batch
            images = batch['image'].to(device)          # (B, 3, H, W)
            heatmaps = batch['heatmap'].to(device)      # (B, H, W)

            # Tambahkan dimensi channel pada heatmap: (B, H, W) -> (B, 1, H, W)
            heatmaps = heatmaps.unsqueeze(1)

            # ---- Forward pass ----
            outputs = model(images)                     # (B, 1, H, W)

            # ---- Hitung loss (MSE antara predicted & ground truth heatmap) ----
            loss = criterion(outputs, heatmaps)

            # ---- Backward pass & optimizer step ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- Hitung MAE (selisih jumlah objek) ----
            # sum(predicted_heatmap) ≈ jumlah objek prediksi
            # sum(ground_truth_heatmap) ≈ jumlah objek sebenarnya
            with torch.no_grad():
                batch_size_actual = images.size(0)
                for i in range(batch_size_actual):
                    pred_count = outputs[i].sum().item()
                    gt_count = heatmaps[i].sum().item()
                    epoch_mae += abs(pred_count - gt_count)

            epoch_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

        # ---- Rata-rata metrik per epoch ----
        avg_loss = epoch_loss / num_samples
        avg_mae = epoch_mae / num_samples
        epoch_time = time.time() - epoch_start

        # ---- Model Checkpointing ----
        # Simpan model jika MAE saat ini lebih baik (lebih kecil)
        is_best = avg_mae < best_mae
        if is_best:
            best_mae = avg_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae,
                'loss': avg_loss,
            }, BEST_MODEL_PATH)
            marker = " ★ SAVED"
        else:
            marker = ""

        # ---- Logging ----
        print(f"  {epoch:>5}  |  {avg_loss:>12.8f}  |  {avg_mae:>10.4f}  |"
              f"  {best_mae:>10.4f}  |  {epoch_time:.1f}s{marker}")

    # ---- Training Selesai ----
    print("-" * 65)
    print(f"\n  Training selesai!")
    print(f"  Best MAE       : {best_mae:.4f}")
    print(f"  Best model     : {os.path.abspath(BEST_MODEL_PATH)}")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    train()
