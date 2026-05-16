import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import arsitektur model dari file proyek
from model_dme import DensityMapRegressor


# ============================================================
# Konfigurasi
# ============================================================
CHECKPOINT_PATH = os.path.join('checkpoints', 'final_dme_97percent.pth')

# Resolusi target – harus sama dengan yang digunakan saat training (dataset_loader.py)
TARGET_SIZE = (672, 512)  # (width, height)

# Normalisasi standar ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def select_device():
    """
    Deteksi device secara otomatis.
    Prioritas: CUDA (Nvidia GPU) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device     : CUDA — {gpu_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"  Device     : MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"  Device     : CPU")
    return device


def get_inference_transforms():
    """
    Pipeline preprocessing untuk inference.
    HANYA Normalize + ToTensorV2, TANPA resize (resize dilakukan secara manual
    agar kita dapat mengembalikan density map ke ukuran asli).
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_model(checkpoint_path, device):
    """
    Inisiasi model, load bobot dari checkpoint, set ke eval mode.
    """
    model = DensityMapRegressor(pretrained=False)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    best_mae = checkpoint.get('best_mae', '?')
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Epoch      : {epoch}")
    print(f"  Best MAE   : {best_mae}")

    return model


def preprocess_image(image_path, transform):
    """
    Baca gambar dengan OpenCV, konversi ke RGB, dan siapkan tensor.
    Menyimpan salinan gambar asli untuk visualisasi dan ukuran asli.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_image = image_rgb.copy()  # untuk visualisasi akhir
    orig_h, orig_w = original_image.shape[:2]

    # Resize ke ukuran training (model hanya melihat resolusi ini saat training)
    image_resized = cv2.resize(image_rgb, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    # Terapkan normalisasi dan konversi ke tensor pada gambar yang sudah di‑resize
    transformed = transform(image=image_resized)
    image_tensor = transformed['image']  # Tensor (C, H, W) – ukuran target

    return image_tensor, original_image, (orig_w, orig_h)


def predict(model, image_tensor, device):
    """
    Jalankan inference pada tensor yang sudah di‑resize ke target size.
    Output density map masih pada ukuran target.
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)  # (1, 1, target_H, target_W)

    # Konversi ke numpy dan hilangkan scaling factor training (1000.0)
    density_map = (output / 1000.0).squeeze().cpu().numpy()  # (target_H, target_W)

    # Hitung prediksi count pada ukuran target (untuk verifikasi)
    predicted_count_target = density_map.sum()

    return density_map, float(predicted_count_target)


def resize_density_map_to_original(density_map, original_size, target_size=TARGET_SIZE):
    """
    Kembalikan density map ke ukuran asli gambar dengan koreksi area.
    
    Parameters:
        density_map (np.ndarray): Density map pada ukuran target.
        original_size (tuple): (orig_w, orig_h) ukuran gambar asli.
        target_size (tuple): (target_w, target_h) ukuran saat inference.
    
    Returns:
        np.ndarray: Density map ukuran asli dengan sum ~ jumlah objek.
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size

    if (orig_w == target_w) and (orig_h == target_h):
        return density_map

    # Simpan sum sebelum resize untuk koreksi area
    sum_before = density_map.sum()

    # Resize menggunakan bilinear interpolation
    density_orig = cv2.resize(density_map, (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)

    # Koreksi area: perkalian dengan (area_target / area_original)
    # Karena setelah resize ke ukuran lebih besar, nilai pixel mengecil.
    area_ratio = (target_w * target_h) / (orig_w * orig_h)
    if sum_before > 0:
        density_orig *= area_ratio

    return density_orig


def create_heatmap_overlay(original_image, density_map_orig):
    """
    Buat overlay heatmap JET di atas gambar asli (ukuran asli).
    """
    # Normalisasi density map ke 0-255
    if density_map_orig.max() > 0:
        density_norm = (density_map_orig / density_map_orig.max() * 255).astype(np.uint8)
    else:
        density_norm = np.zeros_like(density_map_orig, dtype=np.uint8)

    heatmap_bgr = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Ukuran sudah sama, langsung overlay
    overlay = cv2.addWeighted(original_image, 0.5, heatmap_rgb, 0.5, 0)
    return overlay


def visualize_result(original_image, overlay, predicted_count, image_name):
    """
    Tampilkan hasil prediksi menggunakan matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].imshow(original_image)
    axes[0].set_title('Gambar Asli', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title('Density Map Overlay', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    fig.suptitle(
        f'Predicted Count: {predicted_count:.1f}',
        fontsize=22,
        fontweight='bold',
        color='#e74c3c',
        y=0.98,
    )

    fig.text(0.5, 0.01, f'File: {image_name}', ha='center', fontsize=11, color='gray')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()


def run_prediction(image_path, checkpoint_path=CHECKPOINT_PATH, target_size=TARGET_SIZE):
    """
    Pipeline lengkap prediksi yang sekarang scale‑invariant.
    1. Load model
    2. Preprocess: resize gambar ke target size, simpan original
    3. Inference pada ukuran target
    4. Kembalikan density map ke ukuran asli dengan koreksi count
    5. Visualisasi overlay pada gambar asli
    """
    image_name = os.path.basename(image_path)

    print("\n" + "=" * 60)
    print("  PREDICT — Density Map Estimation (DME)  [Scale‑Invariant]")
    print("=" * 60)

    # ---- 1. Device ----
    device = select_device()

    # ---- 2. Load Model ----
    print(f"\n  [Model Loading]")
    if not os.path.exists(checkpoint_path):
        print(f"\n  [ERROR] Checkpoint tidak ditemukan: {checkpoint_path}")
        print(f"  Jalankan training terlebih dahulu: py train.py")
        return
    model = load_model(checkpoint_path, device)

    # ---- 3. Preprocess ----
    print(f"\n  [Preprocessing]")
    print(f"  Image      : {image_path}")
    transform = get_inference_transforms()
    image_tensor, original_image, original_size = preprocess_image(image_path, transform)
    print(f"  Original size : {original_size[0]}x{original_size[1]}")
    print(f"  Resized to    : {target_size[0]}x{target_size[1]} (training resolution)")

    # ---- 4. Inference (pada ukuran target) ----
    print(f"\n  [Inference]")
    density_map_target, count_target = predict(model, image_tensor, device)
    print(f"  Density map (target) shape : {density_map_target.shape}")
    print(f"  Predicted count (target)   : {count_target:.4f}")

    # ---- 5. Kembalikan ke ukuran asli ----
    density_map_orig = resize_density_map_to_original(
        density_map_target, original_size, target_size
    )
    final_count = density_map_orig.sum()
    print(f"  Density map (original) sum : {final_count:.4f}")

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  PREDICTED COUNT : {final_count:>8.1f} objek   │")
    print(f"  └─────────────────────────────────────┘")

    # ---- 6. Visualisasi ----
    overlay = create_heatmap_overlay(original_image, density_map_orig)
    visualize_result(original_image, overlay, final_count, image_name)

    print(f"\n  Prediksi selesai!")
    print("=" * 60 + "\n")

    return final_count


# ============================================================
# Eksekusi Utama
# ============================================================
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        TEST_IMAGE_PATH = sys.argv[1]
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"[ERROR] File gambar tidak ditemukan: {TEST_IMAGE_PATH}")
            sys.exit(1)
        print(f"Menggunakan gambar dari argumen: {TEST_IMAGE_PATH}")
        run_prediction(TEST_IMAGE_PATH)
    else:
        TEST_IMAGE_DIR = os.path.join('dataset', 'images')
        
        supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = sorted([
            f for f in os.listdir(TEST_IMAGE_DIR)
            if f.lower().endswith(supported_ext)
        ])

        if not image_files:
            print(f"Tidak ada gambar di folder '{TEST_IMAGE_DIR}'")
        else:
            TEST_IMAGE_PATH = os.path.join(TEST_IMAGE_DIR, image_files[0])
            print(f"Tidak ada argumen path gambar yang diberikan.")
            print(f"Menggunakan gambar test default: {TEST_IMAGE_PATH}")
            print(f"Tips: Anda bisa menjalankan dengan: py predict.py path/ke/gambar.jpg\n")
            run_prediction(TEST_IMAGE_PATH)