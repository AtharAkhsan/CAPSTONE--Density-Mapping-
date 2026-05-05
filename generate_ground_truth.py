import os
import json
import math
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


# ==============================
# Konfigurasi
# ==============================
IMAGES_DIR = os.path.join('dataset', 'images')
ANNOTATIONS_DIR = os.path.join('dataset', 'annotations')
GROUND_TRUTH_DIR = os.path.join('dataset', 'ground_truth')
BASE_SIGMA = 8  # Sigma dasar pada resolusi referensi

# Resolusi referensi (resolusi training asli)
REFERENCE_RESOLUTION = (672, 512)  # (width, height)


def generate_density_map(image_shape, points, base_sigma=BASE_SIGMA,
                         reference_resolution=REFERENCE_RESOLUTION):
    """
    Generate density map dari list koordinat titik dengan sigma adaptif.

    Sigma dihitung berdasarkan rasio area gambar terhadap resolusi referensi:
        sigma = base_sigma * sqrt((w * h) / (ref_w * ref_h))

    Pada resolusi referensi (672x512), sigma = base_sigma (8).
    Pada resolusi lebih tinggi, sigma meningkat proporsional -> blob lebih besar.
    Pada resolusi lebih rendah, sigma menurun proporsional -> blob lebih kecil.

    Parameters:
        image_shape (tuple): (height, width) dari gambar.
        points (list): List koordinat [[x1, y1], [x2, y2], ...].
        base_sigma (float): Sigma dasar pada resolusi referensi.
        reference_resolution (tuple): (ref_w, ref_h) resolusi referensi.

    Returns:
        numpy.ndarray: Density map (float32) dengan sum ~ jumlah titik.
    """
    h, w = image_shape
    ref_w, ref_h = reference_resolution

    # Adaptive sigma: skala berdasarkan rasio area
    area_ratio = (w * h) / (ref_w * ref_h)
    sigma = base_sigma * math.sqrt(area_ratio)

    density = np.zeros(image_shape, dtype=np.float32)
    num_points = len(points)

    if num_points == 0:
        return density

    points_array = np.array(points)

    # 1. Hitung sigma dinamis menggunakan KDTree
    if num_points > 3:
        tree = KDTree(points_array)
        # Ambil 4 tetangga terdekat (k=4).
        # Hasil pertama adalah titik itu sendiri (jarak 0), jadi ambil k=2,3,4 (indeks 1,2,3)
        distances, _ = tree.query(points_array, k=4)
        
        # Rata-rata jarak ke 3 titik terdekat
        avg_distances = np.mean(distances[:, 1:], axis=1)
        sigmas = beta * avg_distances
        # Clipping sigma
        sigmas = np.clip(sigmas, min_sigma, max_sigma)
    else:
        sigmas = np.full(num_points, default_sigma, dtype=np.float32)

    # 2. Letakkan Gaussian Kernel untuk setiap titik
    for i, point in enumerate(points):
        pt_x, pt_y = int(point[0]), int(point[1])
        sigma = sigmas[i]

        # Batas sebaran kernel (3*sigma mencakup ~99.7% area Gaussian)
        k_size = int(3 * sigma)

        # Buat grid untuk kernel secara independen
        y_grid, x_grid = np.ogrid[-k_size:k_size+1, -k_size:k_size+1]
        H = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))

        H_sum = H.sum()
        if H_sum == 0:
            continue
            
        # Normalisasi kernel agar total nilainya = 1
        H = H / H_sum

        # Tentukan letak kernel pada gambar
        y1, y2 = pt_y - k_size, pt_y + k_size + 1
        x1, x2 = pt_x - k_size, pt_x + k_size + 1

        # Cek jika kernel keluar dari batas gambar, potong kernel-nya
        k_y1, k_y2 = 0, 2 * k_size + 1
        k_x1, k_x2 = 0, 2 * k_size + 1

        if y1 < 0:
            k_y1 = -y1
            y1 = 0
        if y2 > image_shape[0]:
            k_y2 -= (y2 - image_shape[0])
            y2 = image_shape[0]

        if x1 < 0:
            k_x1 = -x1
            x1 = 0
        if x2 > image_shape[1]:
            k_x2 -= (x2 - image_shape[1])
            x2 = image_shape[1]

        # Tambahkan nilai kernel ke area gambar yang valid
        if y1 < y2 and x1 < x2:
            density[y1:y2, x1:x2] += H[k_y1:k_y2, k_x1:k_x2]

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
    print(f"  Annotations          : {ANNOTATIONS_DIR}")
    print(f"  Images               : {IMAGES_DIR}")
    print(f"  Output               : {GROUND_TRUTH_DIR}")
    print(f"  Base sigma           : {BASE_SIGMA}")
    print(f"  Reference resolution : {REFERENCE_RESOLUTION[0]}x{REFERENCE_RESOLUTION[1]}")
    print(f"  Total file           : {len(json_files)}")
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

        # ---- 3. Generate density map (adaptive sigma) ----
        density_map = generate_density_map(
            (h, w), points,
            base_sigma=BASE_SIGMA,
            reference_resolution=REFERENCE_RESOLUTION,
        )

        # Hitung sigma yang digunakan untuk logging
        area_ratio = (w * h) / (REFERENCE_RESOLUTION[0] * REFERENCE_RESOLUTION[1])
        effective_sigma = BASE_SIGMA * math.sqrt(area_ratio)

        print(f"  Effective sigma   : {effective_sigma:.2f}")
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
