import os
import json
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


# ==============================
# Konfigurasi
# ==============================
IMAGES_DIR = os.path.join('dataset', 'images')
ANNOTATIONS_DIR = os.path.join('dataset', 'annotations')
GROUND_TRUTH_DIR = os.path.join('dataset', 'ground_truth')
SIGMA = 15  # Ukuran sebaran Gaussian heatmap


def generate_density_map(image_shape, points, sigma=SIGMA):
    """
    Generate density map dari list koordinat titik.

    Parameters:
        image_shape (tuple): (height, width) dari gambar.
        points (list): List koordinat [[x1, y1], [x2, y2], ...].
        sigma (float): Sigma untuk Gaussian filter.

    Returns:
        numpy.ndarray: Density map (float32).
    """
    density = np.zeros(image_shape, dtype=np.float32)

    for point in points:
        x, y = int(point[0]), int(point[1])

        # Pastikan koordinat dalam batas gambar
        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
            density[y, x] = 1  # Numpy: (baris/y, kolom/x)

    # Aplikasikan Gaussian filter
    if len(points) > 0:
        density = gaussian_filter(density, sigma=sigma)

    return density


def create_visualization(image, density_map):
    """
    Buat overlay heatmap di atas gambar asli.

    Parameters:
        image (numpy.ndarray): Gambar asli (BGR).
        density_map (numpy.ndarray): Density map (float32).

    Returns:
        numpy.ndarray: Gambar overlay (BGR).
    """
    # Normalisasi density map ke 0-255 untuk colormap
    if density_map.max() > 0:
        density_norm = (density_map / density_map.max() * 255).astype(np.uint8)
    else:
        density_norm = np.zeros_like(density_map, dtype=np.uint8)

    # Terapkan colormap JET
    heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)

    # Overlay dengan alpha blending
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    return overlay


def main():
    # Pastikan folder ground_truth ada
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

    # Ambil semua file JSON
    json_files = sorted([
        f for f in os.listdir(ANNOTATIONS_DIR)
        if f.lower().endswith('.json')
    ])

    if not json_files:
        print(f"Tidak ada file .json di folder '{ANNOTATIONS_DIR}'")
        print("Silakan buat anotasi terlebih dahulu menggunakan point_labeler.py")
        return

    print("\n" + "=" * 60)
    print("  GENERATE GROUND TRUTH - Density Map dari Anotasi")
    print("=" * 60)
    print(f"  Annotations : {ANNOTATIONS_DIR}")
    print(f"  Images      : {IMAGES_DIR}")
    print(f"  Output      : {GROUND_TRUTH_DIR}")
    print(f"  Sigma       : {SIGMA}")
    print(f"  Total file  : {len(json_files)}")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for idx, json_file in enumerate(json_files):
        name_without_ext = os.path.splitext(json_file)[0]
        json_path = os.path.join(ANNOTATIONS_DIR, json_file)

        print(f"\n[{idx + 1}/{len(json_files)}] Memproses: {json_file}")

        # ---- 1. Baca file JSON ----
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_filename = data.get('image', f"{name_without_ext}.png")
        points = data.get('points', [])
        num_points = len(points)

        print(f"  Gambar  : {image_filename}")
        print(f"  Jumlah titik : {num_points}")

        # ---- 2. Buka gambar untuk mendapatkan dimensi ----
        image_path = os.path.join(IMAGES_DIR, image_filename)

        if not os.path.exists(image_path):
            print(f"  [ERROR] Gambar tidak ditemukan: {image_path}")
            error_count += 1
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"  [ERROR] Gagal membaca gambar: {image_path}")
            error_count += 1
            continue

        h, w = image.shape[:2]
        print(f"  Dimensi : {w} x {h}")

        # ---- 3. Generate density map ----
        density_map = generate_density_map((h, w), points, sigma=SIGMA)

        print(f"  Density map shape : {density_map.shape}")
        print(f"  Density map max   : {density_map.max():.8f}")
        print(f"  Density map sum   : {density_map.sum():.4f} "
              f"(idealnya ~ {num_points})")

        # ---- 4. Simpan density map sebagai .npy ----
        npy_path = os.path.join(GROUND_TRUTH_DIR, f"{name_without_ext}.npy")
        np.save(npy_path, density_map)
        print(f"  Tersimpan (.npy)  : {npy_path}")

        # ---- 5. Buat dan simpan visualisasi overlay ----
        vis_image = create_visualization(image, density_map)
        vis_path = os.path.join(GROUND_TRUTH_DIR, f"{name_without_ext}_vis.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"  Tersimpan (vis)   : {vis_path}")

        success_count += 1

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  SELESAI!")
    print(f"  Berhasil : {success_count} file")
    if error_count > 0:
        print(f"  Gagal    : {error_count} file")
    print(f"  Output   : {GROUND_TRUTH_DIR}/")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
