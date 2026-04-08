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
CHECKPOINT_PATH = os.path.join('checkpoints', 'best_dme_model.pth')

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
    HANYA Normalize + ToTensorV2, TANPA augmentasi apapun.

    Returns:
        albumentations.Compose: Pipeline transformasi.
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_model(checkpoint_path, device):
    """
    Inisiasi model, load bobot dari checkpoint, set ke eval mode.

    Parameters:
        checkpoint_path (str): Path ke file .pth checkpoint.
        device (torch.device): Device target (cuda/mps/cpu).

    Returns:
        DensityMapRegressor: Model siap inference.
    """
    model = DensityMapRegressor(pretrained=False)
    model = model.to(device)

    # Load state dictionary dari checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set ke evaluation mode (disable dropout, batchnorm pakai running stats)
    model.eval()

    # Info checkpoint
    epoch = checkpoint.get('epoch', '?')
    best_mae = checkpoint.get('best_mae', '?')
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Epoch      : {epoch}")
    print(f"  Best MAE   : {best_mae}")

    return model


def preprocess_image(image_path, transform):
    """
    Baca gambar dengan OpenCV, konversi ke RGB, dan terapkan preprocessing.

    Parameters:
        image_path (str): Path ke file gambar.
        transform (albumentations.Compose): Pipeline transformasi.

    Returns:
        tuple: (image_tensor, image_original)
            - image_tensor: Tensor (3, H, W) ternormalisasi, siap masuk model
            - image_original: Numpy array (H, W, 3) RGB, untuk visualisasi
    """
    # Baca gambar (BGR) dan konversi ke RGB
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Simpan versi asli (tanpa resize) untuk visualisasi
    image_display = image_rgb.copy()

    # Terapkan preprocessing (Normalize + ToTensor)
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image']  # Tensor (C, H, W)

    return image_tensor, image_display


def predict(model, image_tensor, device):
    """
    Jalankan inference pada satu gambar.

    Parameters:
        model (DensityMapRegressor): Model dalam eval mode.
        image_tensor (torch.Tensor): Input tensor (3, H, W).
        device (torch.device): Device target.

    Returns:
        tuple: (density_map, predicted_count)
            - density_map: Numpy array (H, W) float32
            - predicted_count: float, estimasi jumlah objek
    """
    # Tambahkan batch dimension: (3, H, W) -> (1, 3, H, W)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Inference tanpa menghitung gradient
    with torch.no_grad():
        output = model(image_tensor)  # (1, 1, H, W)

    # Konversi output ke numpy
    density_map = output.squeeze().cpu().numpy()  # (H, W)

    # Hitung jumlah objek = sum seluruh piksel density map
    predicted_count = density_map.sum()

    return density_map, float(predicted_count)


def create_heatmap_overlay(image_display, density_map):
    """
    Buat overlay heatmap JET di atas gambar asli.

    Parameters:
        image_display (numpy.ndarray): Gambar RGB (H, W, 3) untuk tampilan.
        density_map (numpy.ndarray): Density map (H, W) float32.

    Returns:
        numpy.ndarray: Gambar overlay RGB (H, W, 3).
    """
    # Normalisasi density map ke 0-255
    if density_map.max() > 0:
        density_norm = (density_map / density_map.max() * 255).astype(np.uint8)
    else:
        density_norm = np.zeros_like(density_map, dtype=np.uint8)

    # Terapkan colormap JET (output BGR)
    heatmap_bgr = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Resize heatmap agar sesuai dengan gambar display
    heatmap_rgb = cv2.resize(heatmap_rgb, (image_display.shape[1], image_display.shape[0]))

    # Overlay: 50% gambar asli + 50% heatmap
    overlay = cv2.addWeighted(image_display, 0.5, heatmap_rgb, 0.5, 0)

    return overlay


def visualize_result(image_display, overlay, predicted_count, image_name):
    """
    Tampilkan hasil prediksi menggunakan matplotlib.

    Parameters:
        image_display (numpy.ndarray): Gambar asli RGB.
        overlay (numpy.ndarray): Gambar overlay heatmap RGB.
        predicted_count (float): Estimasi jumlah objek.
        image_name (str): Nama file gambar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel Kiri: Gambar Asli
    axes[0].imshow(image_display)
    axes[0].set_title('Gambar Asli', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Panel Kanan: Heatmap Overlay
    axes[1].imshow(overlay)
    axes[1].set_title('Density Map Overlay', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Judul utama: Predicted Count
    fig.suptitle(
        f'Predicted Count: {predicted_count:.1f}',
        fontsize=22,
        fontweight='bold',
        color='#e74c3c',
        y=0.98,
    )

    # Subtitle: nama file
    fig.text(0.5, 0.01, f'File: {image_name}', ha='center', fontsize=11, color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()


def run_prediction(image_path, checkpoint_path=CHECKPOINT_PATH):
    """
    Pipeline lengkap prediksi: load model → preprocess → predict → visualize.

    Parameters:
        image_path (str): Path ke gambar yang ingin diprediksi.
        checkpoint_path (str): Path ke file checkpoint model.
    """
    image_name = os.path.basename(image_path)

    print("\n" + "=" * 60)
    print("  PREDICT — Density Map Estimation (DME)")
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
    image_tensor, image_display = preprocess_image(image_path, transform)
    print(f"  Tensor     : {image_tensor.shape}")

    # ---- 4. Inference ----
    print(f"\n  [Inference]")
    density_map, predicted_count = predict(model, image_tensor, device)
    print(f"  Density map shape : {density_map.shape}")
    print(f"  Density map max   : {density_map.max():.6f}")
    print(f"  Density map sum   : {density_map.sum():.4f}")
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  PREDICTED COUNT : {predicted_count:>8.1f} objek   │")
    print(f"  └─────────────────────────────────────┘")

    # ---- 5. Visualisasi ----
    overlay = create_heatmap_overlay(image_display, density_map)
    visualize_result(image_display, overlay, predicted_count, image_name)

    print(f"\n  Prediksi selesai!")
    print("=" * 60 + "\n")

    return predicted_count


# ============================================================
# Eksekusi Utama
# ============================================================
if __name__ == '__main__':
    # Default: gunakan gambar pertama di dataset/images/
    TEST_IMAGE_DIR = os.path.join('dataset', 'images')

    # Cari gambar pertama yang tersedia
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = sorted([
        f for f in os.listdir(TEST_IMAGE_DIR)
        if f.lower().endswith(supported_ext)
    ])

    if not image_files:
        print(f"Tidak ada gambar di folder '{TEST_IMAGE_DIR}'")
    else:
        TEST_IMAGE_PATH = os.path.join(TEST_IMAGE_DIR, image_files[0])
        print(f"Menggunakan gambar test: {TEST_IMAGE_PATH}")
        run_prediction(TEST_IMAGE_PATH)
 