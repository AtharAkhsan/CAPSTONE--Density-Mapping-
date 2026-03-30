import cv2
import os
import json
import copy


# ==============================
# Konfigurasi
# ==============================
IMAGES_DIR = os.path.join('dataset', 'images')
ANNOTATIONS_DIR = os.path.join('dataset', 'annotations')
POINT_COLOR = (0, 255, 0)    # Hijau (BGR)
POINT_RADIUS = 5
POINT_THICKNESS = -1          # Filled circle
WINDOW_NAME = 'Point Labeler'
MAX_DISPLAY_SIZE = 900        # Maksimal lebar/tinggi window


# ==============================
# Variabel Global
# ==============================
points = []
display_image = None
original_image = None
scale_factor = 1.0


def mouse_callback(event, x, y, flags, param):
    """Callback untuk mendeteksi klik kiri mouse dan menyimpan koordinat."""
    global points, display_image

    if event == cv2.EVENT_LBUTTONDOWN:
        # Konversi koordinat display ke koordinat gambar asli
        orig_x = int(x / scale_factor)
        orig_y = int(y / scale_factor)

        points.append([orig_x, orig_y])

        # Gambar titik hijau di lokasi klik (pada display image)
        cv2.circle(display_image, (x, y), POINT_RADIUS, POINT_COLOR, POINT_THICKNESS)
        cv2.imshow(WINDOW_NAME, display_image)

        print(f"  + Titik ditambahkan: ({orig_x}, {orig_y})  |  Total: {len(points)} titik")


def redraw_image():
    """Redraw semua titik pada gambar (digunakan setelah undo)."""
    global display_image
    display_image = get_display_image(original_image)

    for point in points:
        # Konversi koordinat asli ke koordinat display
        disp_x = int(point[0] * scale_factor)
        disp_y = int(point[1] * scale_factor)
        cv2.circle(display_image, (disp_x, disp_y), POINT_RADIUS, POINT_COLOR, POINT_THICKNESS)

    cv2.imshow(WINDOW_NAME, display_image)


def get_display_image(image):
    """Resize gambar jika terlalu besar untuk layar."""
    global scale_factor
    h, w = image.shape[:2]
    scale_factor = 1.0

    if max(h, w) > MAX_DISPLAY_SIZE:
        scale_factor = MAX_DISPLAY_SIZE / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image.copy()


def print_instructions():
    """Tampilkan instruksi penggunaan di terminal."""
    print("\n" + "=" * 50)
    print("  POINT LABELER - Tool Anotasi Titik Koordinat")
    print("=" * 50)
    print("  Klik Kiri  : Tandai titik pada gambar")
    print("  Tekan 'z'  : Undo (hapus titik terakhir)")
    print("  Tekan 's'  : Simpan anotasi ke file .json")
    print("  Tekan 'd'  : Lanjut ke gambar berikutnya")
    print("  Tekan 'q'  : Keluar dari program")
    print("=" * 50)


def main():
    global points, display_image, original_image

    # Pastikan folder annotations ada
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    # Ambil semua file gambar dari folder images
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(supported_ext)
    ])

    if not image_files:
        print(f"Tidak ada gambar ditemukan di folder '{IMAGES_DIR}'")
        print(f"Silakan tambahkan gambar terlebih dahulu.")
        return

    print_instructions()
    print(f"\nDitemukan {len(image_files)} gambar di '{IMAGES_DIR}'\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    for idx, filename in enumerate(image_files):
        image_path = os.path.join(IMAGES_DIR, filename)
        name_without_ext = os.path.splitext(filename)[0]
        json_path = os.path.join(ANNOTATIONS_DIR, f"{name_without_ext}.json")

        # Reset points untuk gambar baru
        points = []

        # Load gambar asli
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"  Gagal membaca gambar: {image_path}, skip...")
            continue

        h, w = original_image.shape[:2]

        # Cek apakah sudah ada anotasi sebelumnya
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                points = data.get('points', [])
            print(f"\n[{idx + 1}/{len(image_files)}] {filename} ({w}x{h}) "
                  f"- Loaded {len(points)} titik dari anotasi sebelumnya")
        else:
            print(f"\n[{idx + 1}/{len(image_files)}] {filename} ({w}x{h}) "
                  f"- Belum ada anotasi")

        # Tampilkan gambar dengan titik yang sudah ada
        redraw_image()

        while True:
            key = cv2.waitKey(0) & 0xFF

            # ---- Undo: hapus titik terakhir ----
            if key == ord('z'):
                if points:
                    removed = points.pop()
                    print(f"  - Undo titik: ({removed[0]}, {removed[1]})  "
                          f"|  Sisa: {len(points)} titik")
                    redraw_image()
                else:
                    print("  ! Tidak ada titik untuk di-undo")

            # ---- Save: simpan ke JSON ----
            elif key == ord('s'):
                annotation_data = {
                    'image': filename,
                    'image_width': w,
                    'image_height': h,
                    'count': len(points),
                    'points': copy.deepcopy(points)
                }
                with open(json_path, 'w') as f:
                    json.dump(annotation_data, f, indent=2)
                print(f"  >>> Tersimpan: {json_path} ({len(points)} titik)")

            # ---- Next: lanjut ke gambar berikutnya ----
            elif key == ord('d'):
                print(f"  >> Lanjut ke gambar berikutnya...")
                break

            # ---- Quit: keluar dari program ----
            elif key == ord('q'):
                print("\n  Keluar dari Point Labeler. Bye!")
                cv2.destroyAllWindows()
                return

    print("\n" + "=" * 50)
    print("  Semua gambar sudah selesai di-label!")
    print("=" * 50)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
