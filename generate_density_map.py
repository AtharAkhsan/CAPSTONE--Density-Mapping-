import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def points_to_density_map(image_shape, points):
    """
    Mengonversi list koordinat titik menjadi density map menggunakan Gaussian filter.

    Parameters:
        image_shape (tuple): Ukuran gambar dalam format (height, width).
        points (list): List koordinat titik dalam format [[x1, y1], [x2, y2], ...].

    Returns:
        numpy.ndarray: Density map hasil konvolusi Gaussian.
    """
    # Buat matriks nol dengan ukuran image_shape
    density_map = np.zeros(image_shape, dtype=np.float32)

    # Letakkan nilai 1 pada setiap koordinat titik
    for point in points:
        x, y = int(point[0]), int(point[1])

        # Pastikan koordinat berada di dalam batas gambar
        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
            density_map[y, x] = 1

    # Aplikasikan Gaussian filter dengan sigma=4
    density_map = gaussian_filter(density_map, sigma=4)

    return density_map


def visualize_heatmap(image_path, density_map):
    """
    Menampilkan overlay density map (heatmap) di atas gambar asli.

    Parameters:
        image_path (str): Path ke file gambar asli.
        density_map (numpy.ndarray): Matriks density map hasil generate.
    """
    # Load gambar asli menggunakan OpenCV dan konversi BGR -> RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Tampilkan gambar asli dengan overlay density map
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.imshow(density_map, cmap='jet', alpha=0.5)
    plt.colorbar(label='Density')
    plt.title('Density Map Overlay')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import os
    import random

    # Path ke folder images
    images_dir = os.path.join('dataset', 'images')

    # Ambil file gambar pertama dari folder dataset/images/
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        print("Tidak ada gambar di folder dataset/images/. "
              "Silakan tambahkan gambar terlebih dahulu.")
    else:
        image_path = os.path.join(images_dir, image_files[0])
        print(f"Menggunakan gambar: {image_path}")

        # Load gambar untuk mendapatkan ukuran (height, width)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        print(f"Ukuran gambar: {w} x {h}")

        # Buat dummy 5 titik koordinat [x, y] secara acak
        dummy_points = [[random.randint(0, w - 1), random.randint(0, h - 1)]
                        for _ in range(5)]
        print(f"Dummy points: {dummy_points}")

        # Generate density map
        density = points_to_density_map((h, w), dummy_points)
        print(f"Density map shape: {density.shape}")
        print(f"Density map max value: {density.max():.6f}")

        # Visualisasi heatmap overlay
        visualize_heatmap(image_path, density)
