import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==============================
# Konfigurasi Default
# ==============================
IMAGES_DIR = os.path.join('dataset', 'images')
GROUND_TRUTH_DIR = os.path.join('dataset', 'ground_truth')

# Normalisasi standar ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """
    Pipeline augmentasi untuk training.

    Spatial transforms (berlaku untuk image & mask):
        - HorizontalFlip, VerticalFlip

    Pixel-level transforms (HANYA berlaku untuk image):
        - RandomSunFlare (simulasi silau cahaya pada plastik)
        - RandomBrightnessContrast

    Returns:
        albumentations.Compose: Pipeline augmentasi.
    """
    return A.Compose([

        # ---- Spatial Transforms (image & mask) ----
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # ---- Pixel-level Transforms (HANYA image) ----
        A.RandomSunFlare(
            p=0.2,
            src_radius=100,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3,
        ),

        # ---- Normalisasi ImageNet & konversi ke Tensor ----
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    Pipeline transformasi untuk validasi/testing (tanpa augmentasi).

    Returns:
        albumentations.Compose: Pipeline transformasi.
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class DMEDataset(Dataset):
    """
    PyTorch Custom Dataset untuk Density Map Estimation.

    Membaca pasangan gambar RGB dan heatmap numpy (.npy) yang namanya
    saling berkesesuaian (misal: img_0001.jpg <-> img_0001.npy).

    Parameters:
        images_dir (str): Path ke folder gambar.
        ground_truth_dir (str): Path ke folder ground truth (.npy).
        transform (albumentations.Compose): Pipeline augmentasi/transformasi.
    """

    def __init__(self, images_dir=IMAGES_DIR, ground_truth_dir=GROUND_TRUTH_DIR,
                 transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform

        # Kumpulkan semua file .npy (ground truth) yang tersedia
        npy_files = sorted(glob.glob(os.path.join(ground_truth_dir, '*.npy')))

        # Cocokkan dengan gambar yang ada di images_dir
        self.pairs = []
        for npy_path in npy_files:
            basename = os.path.splitext(os.path.basename(npy_path))[0]

            # Cari gambar dengan nama yang sama (berbagai ekstensi)
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                candidate = os.path.join(images_dir, basename + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break

            if image_path is not None:
                self.pairs.append({
                    'image_path': image_path,
                    'gt_path': npy_path,
                    'name': basename,
                })

        print(f"[DMEDataset] Ditemukan {len(self.pairs)} pasangan "
              f"(image + ground truth)")

        if len(self.pairs) == 0:
            print(f"  [WARNING] Tidak ada pasangan data yang ditemukan!")
            print(f"  Images dir     : {os.path.abspath(images_dir)}")
            print(f"  Ground truth   : {os.path.abspath(ground_truth_dir)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Mengambil satu pasangan data (image + heatmap).

        Returns:
            dict: {
                'image': Tensor (3, H, W) — gambar ternormalisasi ImageNet,
                'heatmap': Tensor (H, W) — density map float32,
                'name': str — nama file (tanpa ekstensi),
            }
        """
        pair = self.pairs[idx]

        # ---- Load gambar (BGR -> RGB) ----
        image = cv2.imread(pair['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ---- Load heatmap (.npy) ----
        heatmap = np.load(pair['gt_path']).astype(np.float32)

        # ---- Terapkan augmentasi/transformasi ----
        if self.transform is not None:
            # Albumentations: image & mask ditransformasi secara sinkron
            transformed = self.transform(image=image, mask=heatmap)
            image = transformed['image']       # Tensor (C, H, W)
            heatmap = transformed['mask']       # Tensor (H, W)

        # Pastikan heatmap bertipe float32
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap).float()
        else:
            heatmap = heatmap.float()

        return {
            'image': image,
            'heatmap': heatmap,
            'name': pair['name'],
        }


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalisasi tensor gambar dari normalisasi ImageNet
    agar bisa divisualisasikan dengan benar.

    Parameters:
        tensor (torch.Tensor): Tensor (C, H, W) ternormalisasi.
        mean (list): Mean ImageNet.
        std (list): Std ImageNet.

    Returns:
        numpy.ndarray: Gambar (H, W, C) dalam range [0, 1].
    """
    img = tensor.clone().detach().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    # (C, H, W) -> (H, W, C)
    return img.permute(1, 2, 0).numpy()


# ==============================
# Sanity Check
# ==============================
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("  DATASET LOADER - Sanity Check")
    print("=" * 60)

    # Buat dataset dengan augmentasi training
    train_transform = get_train_transforms()
    dataset = DMEDataset(
        images_dir=IMAGES_DIR,
        ground_truth_dir=GROUND_TRUTH_DIR,
        transform=train_transform,
    )

    if len(dataset) == 0:
        print("\n  Tidak ada data untuk di-load. Pastikan sudah menjalankan:")
        print("  1. py point_labeler.py    (anotasi titik)")
        print("  2. py generate_ground_truth.py  (generate .npy)")
        exit()

    # Load 1 sample
    sample = dataset[0]
    image_tensor = sample['image']
    heatmap_tensor = sample['heatmap']
    name = sample['name']

    print(f"\n  Sample: {name}")
    print(f"  Image tensor shape  : {image_tensor.shape}")
    print(f"  Image tensor dtype  : {image_tensor.dtype}")
    print(f"  Heatmap tensor shape: {heatmap_tensor.shape}")
    print(f"  Heatmap tensor dtype: {heatmap_tensor.dtype}")
    print(f"  Heatmap sum (count) : {heatmap_tensor.sum().item():.4f}")
    print(f"  Heatmap max         : {heatmap_tensor.max().item():.8f}")

    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(f"\n  [DataLoader] Batch image shape  : {batch['image'].shape}")
    print(f"  [DataLoader] Batch heatmap shape: {batch['heatmap'].shape}")

    # ---- Visualisasi ----
    # Denormalisasi gambar untuk tampilan
    img_display = denormalize(image_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Gambar setelah augmentasi (denormalisasi)
    axes[0].imshow(img_display)
    axes[0].set_title(f'Augmented Image\n{name}', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Heatmap (density map)
    hm = heatmap_tensor.numpy()
    im = axes[1].imshow(hm, cmap='jet')
    axes[1].set_title(
        f'Density Map (Heatmap)\nsum={hm.sum():.2f}',
        fontsize=12, fontweight='bold'
    )
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(img_display)
    axes[2].imshow(hm, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle('DMEDataset Sanity Check', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    print("\n  Sanity check selesai!")
    print("=" * 60 + "\n")
