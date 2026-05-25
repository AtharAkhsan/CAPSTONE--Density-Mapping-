import os
import time
import argparse
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
# Hyperparameters Default
# ============================================================
EPOCHS = 30
BATCH_SIZE = 2          # Kecil karena komputasi heatmap cukup berat
LEARNING_RATE_SCRATCH = 5e-5    # Learning rate untuk training dari scratch
LEARNING_RATE_FINETUNE = 1e-5   # Learning rate lebih kecil untuk fine-tuning
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


def train(args):
    """
    Fungsi utama untuk training model DME.
    Bisa training dari scratch (ImageNet pretrained weights) atau fine-tuning
    dari external checkpoint.
    """
    print("\n" + "=" * 65)
    print("  TRAINING / FINE-TUNING — Density Map Estimation (DME)")
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
        print("  Pastikan sudah menjalankan anotasi dan generate_ground_truth.py.")
        return

    train_samples, val_samples = stratified_split(all_samples, val_ratio=0.2)

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
        scale_range=(1.0, 1.0),
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

    # ---- 4. Model & Checkpoint Loading ----
    print(f"\n  [Model]")
    model = DensityMapRegressor(pretrained=not args.resume)
    
    # Tentukan learning rate berdasarkan mode (resume vs scratch)
    current_lr = args.lr if args.lr is not None else (LEARNING_RATE_FINETUNE if args.resume else LEARNING_RATE_SCRATCH)
    start_epoch = 1

    if args.resume:
        if not os.path.exists(args.resume):
            print(f"  [ERROR] Checkpoint resume tidak ditemukan: {args.resume}")
            return
        
        print(f"  Resuming from  : {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'epoch' in checkpoint:
            print(f"  Loaded epoch   : {checkpoint['epoch']}")
        if 'best_mae' in checkpoint or 'best_val_mae' in checkpoint:
            best_mae_val = checkpoint.get('best_val_mae', checkpoint.get('best_mae', '?'))
            print(f"  Loaded Best MAE: {best_mae_val}")
    else:
        print(f"  Training from  : scratch (ImageNet pretrained backbone)")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture   : MobileNetV2 + Dilated Conv + Upsample")
    print(f"  Total params   : {total_params:,}")

    # ---- 5. Loss Function & Optimizer ----
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=current_lr)

    if args.resume and args.load_optimizer:
        # User requested to restore optimizer state. Requires weights_only=False
        checkpoint_full = torch.load(args.resume, map_location=device, weights_only=False) 
        if 'optimizer_state_dict' in checkpoint_full:
            optimizer.load_state_dict(checkpoint_full['optimizer_state_dict'])
            print(f"  Optimizer state: Restored from checkpoint")
            
            # Update learning rate on restored optimizer if user specified a custom lr or to ensure finetune lr is used
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            print(f"  Optimizer state: Not found in checkpoint, starting fresh.")

    print(f"\n  [Training Config]")
    print(f"  Loss function  : MSELoss")
    print(f"  Optimizer      : Adam")
    print(f"  Learning rate  : {current_lr}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Freeze backbone: {'Yes (first 5 epochs)' if args.freeze_backbone else 'No'}")
    print(f"  Checkpoint dir : {os.path.abspath(CHECKPOINT_DIR)}")

    # ---- 6. Training Loop ----
    best_val_mae = float('inf')

    print("\n" + "-" * 80)
    print(f"  {'Epoch':>5}  |  {'Train Loss':>12}  |  {'Train MAE':>10}  |  "
          f"{'Val MAE':>10}  |  {'Best Val MAE':>12}  |  Time")
    print("-" * 80)

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        # ======== FREEZE / UNFREEZE LOGIC ========
        if args.freeze_backbone:
            if epoch <= 5:
                # Freeze features
                for param in model.features.parameters():
                    param.requires_grad = False
                if epoch == 1:
                    print("  [INFO] Backbone is FROZEN for the first 5 epochs.")
            elif epoch == 6:
                # Unfreeze features
                for param in model.features.parameters():
                    param.requires_grad = True
                print("  [INFO] Backbone UNFROZEN for the remaining epochs.")

        # ======== TRAINING PHASE ========
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        num_train_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)          # (B, 3, H, W)
            heatmaps = batch['heatmap'].to(device)      # (B, H, W)
            heatmaps = heatmaps.unsqueeze(1)

            outputs = model(images)                     # (B, 1, H, W)
            loss = criterion(outputs, heatmaps * 1000.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                    gt_count = heatmaps[i].sum().item()
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
    parser = argparse.ArgumentParser(description="DME Training & Fine-tuning Script")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume or fine-tune from")
    parser.add_argument("--freeze-backbone", action="store_true", 
                        help="Freeze the MobileNetV2 backbone for the first 5 epochs")
    parser.add_argument("--load-optimizer", action="store_true", 
                        help="Load optimizer state from checkpoint (if resuming)")
    parser.add_argument("--lr", type=float, default=None, 
                        help="Override learning rate (default: 5e-5 scratch, 1e-5 fine-tune)")
    
    args = parser.parse_args()
    train(args)
