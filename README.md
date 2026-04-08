# 🔬 Density Map Estimation for Industrial Part Counting

> **Capstone Project** — Estimasi jumlah objek (baut, mur, komponen industri) menggunakan Density Map Regression berbasis Deep Learning.

Proyek ini membangun pipeline end-to-end untuk menghitung jumlah part/komponen industri dalam sebuah gambar tanpa perlu mendeteksi setiap objek secara individual. Pendekatan yang digunakan adalah **Density Map Estimation**, di mana model memprediksi sebuah continuous heatmap dan jumlah objek diperoleh dari integrasi (penjumlahan piksel) density map tersebut.

---

## 📋 Daftar Isi

- [Arsitektur Proyek](#-arsitektur-proyek)
- [Struktur Folder](#-struktur-folder)
- [Prasyarat & Instalasi](#-prasyarat--instalasi)
- [Pipeline Penggunaan](#-pipeline-penggunaan)
  - [Step 1: Persiapan Gambar](#step-1-persiapan-gambar)
  - [Step 2: Anotasi Titik Koordinat](#step-2-anotasi-titik-koordinat-point_labelerpy)
  - [Step 3: Generate Ground Truth](#step-3-generate-ground-truth-density-map-generate_ground_truthpy)
  - [Step 4: Arsitektur Model](#step-4-arsitektur-model-model_dmepy)
  - [Step 5: Utilitas & Visualisasi](#step-5-utilitas--visualisasi-generate_density_mappy)
- [Detail Teknis](#-detail-teknis)
  - [Density Map Generation](#density-map-generation)
  - [Model Architecture](#model-architecture)
  - [Format Anotasi](#format-anotasi-json)
- [Cara Menjalankan](#-cara-menjalankan)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Roadmap & Pengembangan](#-roadmap--pengembangan)

---

## 🏗 Arsitektur Proyek

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Gambar Asli    │────▶│ Point Labeler│────▶│ Anotasi (.json)  │
│  (dataset/      │     │ (GUI Tool)   │     │ (dataset/        │
│   images/)      │     └──────────────┘     │  annotations/)   │
└─────────────────┘                          └────────┬─────────┘
                                                      │
                                                      ▼
                                             ┌──────────────────┐
                                             │ Generate Ground  │
                                             │ Truth Script     │
                                             └────────┬─────────┘
                                                      │
                                    ┌─────────────────┴────────────────┐
                                    ▼                                  ▼
                           ┌────────────────┐              ┌───────────────────┐
                           │ Density Map    │              │ Visualisasi       │
                           │ (.npy)         │              │ (_vis.jpg)        │
                           │ (ground_truth/)│              │ (ground_truth/)   │
                           └───────┬────────┘              └───────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ DensityMap      │
                          │ Regressor       │
                          │ (MobileNetV2 +  │
                          │  Dilated Conv)  │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ Predicted       │
                          │ Density Map     │
                          │ (sum = count)   │
                          └─────────────────┘
```

---

## 📁 Struktur Folder

```
CAPSTONE (Density Mapping)/
│
├── 📄 README.md                    # Dokumentasi proyek ini
│
├── 🐍 point_labeler.py             # GUI tool anotasi titik koordinat
├── 🐍 generate_ground_truth.py     # Script generate density map ground truth
├── 🐍 generate_density_map.py      # Utilitas & visualisasi density map
├── 🐍 model_dme.py                 # Arsitektur model deep learning
├── 🐍 dataset_loader.py            # PyTorch Dataset & augmentasi (Albumentations)
├── 🐍 train.py                     # Script utama proses training dengan MAE
├── 🐍 predict.py                   # Script inference / prediksi pada gambar baru
│
└── 📂 dataset/
    ├── 📂 images/                   # Foto asli baut/part (.png, .jpg, .bmp)
    │   └── sample_bolts.png
    │
    ├── 📂 annotations/              # File koordinat titik (.json)
    │   └── sample_bolts.json
    │
    └── 📂 ground_truth/             # Hasil generate density map
        ├── sample_bolts.npy         # Density map (numpy array, float32)
        └── sample_bolts_vis.jpg     # Visualisasi overlay heatmap
```

---

## ⚙ Prasyarat & Instalasi

### System Requirements

- **Python** 3.8+
- **OS**: Windows / Linux / macOS
- **GPU** (opsional, untuk training model): CUDA-compatible NVIDIA GPU

### Instalasi Dependencies

```bash
# Install semua dependencies sekaligus
pip install numpy opencv-python scipy matplotlib torch torchvision albumentations
```

| Library | Versi Min. | Kegunaan |
|---------|-----------|----------|
| `numpy` | 1.21+ | Operasi matriks & array |
| `opencv-python` | 4.5+ | Image processing & GUI labeler |
| `scipy` | 1.7+ | Gaussian filter untuk density map |
| `matplotlib` | 3.4+ | Visualisasi heatmap overlay |
| `torch` | 2.0+ | Deep learning framework |
| `torchvision` | 0.15+ | Pretrained MobileNetV2 |
| `albumentations` | 1.3+ | Pipeline augmentasi gambar dan density map secara sinkron |

> **Catatan Windows:** Jika `python` tidak dikenali, gunakan `py` sebagai gantinya (Python Launcher for Windows).

---

## 🚀 Pipeline Penggunaan

### Step 1: Persiapan Gambar

Letakkan semua foto baut/part yang ingin dihitung ke dalam folder:

```
dataset/images/
```

Format yang didukung: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

---

### Step 2: Anotasi Titik Koordinat (`point_labeler.py`)

Tool GUI interaktif untuk menandai lokasi setiap objek pada gambar.

```bash
py point_labeler.py
```

#### Kontrol Keyboard

| Tombol | Fungsi |
|--------|--------|
| **Klik Kiri** | Tandai titik pada objek (muncul dot hijau) |
| **`z`** | Undo — hapus titik terakhir |
| **`s`** | Simpan anotasi ke `dataset/annotations/<nama>.json` |
| **`d`** | Lanjut ke gambar berikutnya |
| **`q`** | Keluar dari program |

#### Fitur

- ✅ Auto-resize gambar besar (max 900px) agar muat di layar
- ✅ Koordinat tetap disimpan pada resolusi asli gambar
- ✅ Load anotasi sebelumnya secara otomatis jika sudah pernah di-save
- ✅ Progres ditampilkan di terminal (`[1/N] filename.jpg`)

#### Output

File JSON disimpan di `dataset/annotations/` dengan format:

```json
{
  "image": "sample_bolts.png",
  "image_width": 1024,
  "image_height": 1024,
  "count": 7,
  "points": [
    [915, 514],
    [782, 513],
    [669, 427]
  ]
}
```

---

### Step 3: Generate Ground Truth Density Map (`generate_ground_truth.py`)

Mengonversi anotasi titik menjadi density map (ground truth) untuk training model.

```bash
py generate_ground_truth.py
```

#### Proses Internal

1. Membaca semua `.json` dari `dataset/annotations/`
2. Membuka gambar asli untuk mendapatkan dimensi (H × W)
3. Membuat matriks nol berukuran H × W
4. Meletakkan nilai `1` pada setiap koordinat `(y, x)` dari anotasi
5. Mengaplikasikan **Gaussian filter** dengan `sigma=15`
6. Menyimpan hasil sebagai `.npy` (presisi float32) dan `_vis.jpg` (visualisasi)

#### Output

| File | Format | Keterangan |
|------|--------|------------|
| `<nama>.npy` | NumPy float32 | Density map dengan presisi penuh |
| `<nama>_vis.jpg` | JPEG | Overlay heatmap JET di atas gambar asli |

#### Terminal Output

```
============================================================
  GENERATE GROUND TRUTH - Density Map dari Anotasi
============================================================
  Annotations : dataset\annotations
  Images      : dataset\images
  Output      : dataset\ground_truth
  Sigma       : 15
  Total file  : 1
============================================================

[1/1] Memproses: sample_bolts.json
  Gambar  : sample_bolts.png
  Jumlah titik : 7
  Dimensi : 1024 x 1024
  Density map shape : (1024, 1024)
  Density map max   : 0.00070736
  Density map sum   : 7.0000 (idealnya ~ 7)
  Tersimpan (.npy)  : dataset\ground_truth\sample_bolts.npy
  Tersimpan (vis)   : dataset\ground_truth\sample_bolts_vis.jpg

============================================================
  SELESAI!
  Berhasil : 1 file
  Output   : dataset\ground_truth/
============================================================
```

> **Catatan:** Nilai `Density map sum` harus mendekati jumlah titik anotasi. Ini membuktikan bahwa Gaussian filter mempertahankan integritas jumlah objek.

---

### Step 4: Arsitektur Model (`model_dme.py`)

Model deep learning untuk memprediksi density map dari gambar input.

```bash
# Quick test arsitektur model
py model_dme.py
```

#### Arsitektur: `DensityMapRegressor`

```
Input Image (3, 224, 224)
         │
         ▼
┌─────────────────────────┐
│  MobileNetV2 (Pretrained)│   Feature Extractor
│  Output: (1280, 7, 7)   │   (Classifier dihapus)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Dilated Conv2D Layer 1   │   1280 → 512 ch, dilation=2
│ + BatchNorm + ReLU       │
├─────────────────────────┤
│ Dilated Conv2D Layer 2   │   512 → 128 ch, dilation=2
│ + BatchNorm + ReLU       │
├─────────────────────────┤
│ Dilated Conv2D Layer 3   │   128 → 1 ch, dilation=2
│ + ReLU (non-negatif)     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Bilinear Upsample (32x)  │   (1, 7, 7) → (1, 224, 224)
└────────────┬────────────┘
             │
             ▼
    Density Map (1, 224, 224)
    sum(pixels) ≈ object count
```

#### Spesifikasi Model

| Property | Value |
|----------|-------|
| Backbone | MobileNetV2 (pretrained on ImageNet) |
| Dilated Conv Layers | 3 layers, dilation=2 |
| Upsample | Bilinear, scale_factor=32 |
| Input Shape | `(batch, 3, 224, 224)` |
| Output Shape | `(batch, 1, 224, 224)` |
| Total Parameters | 8,715,009 |
| Aktivasi Akhir | ReLU (output ≥ 0) |

#### Mengapa Dilated Convolution?

Dilated convolution memperluas **receptive field** tanpa menambah jumlah parameter. Ini penting untuk:
- Menangkap konteks spasial yang lebih luas
- Mengenali objek yang saling **tumpang tindih** atau berdekatan
- Membedakan objek individual dalam area yang padat

---

### Step 5: Utilitas & Visualisasi (`generate_density_map.py`)

Modul utilitas berisi fungsi-fungsi yang dapat digunakan kembali (reusable).

```bash
# Demo dengan dummy data
py generate_density_map.py
```

#### Fungsi yang Tersedia

| Fungsi | Deskripsi |
|--------|-----------|
| `points_to_density_map(image_shape, points)` | Konversi koordinat titik → density map (sigma=4) |
| `visualize_heatmap(image_path, density_map)` | Overlay density map di atas gambar (matplotlib, cmap='jet') |

---

## 🔍 Detail Teknis

### Density Map Generation

Proses mengonversi anotasi titik menjadi density map:

```
Anotasi Titik          Matriks Delta           Density Map (Gaussian)
                       
  • (x1, y1)    →     0 0 0 0 0 0      →     0.0 0.1 0.3 0.1 0.0
  • (x2, y2)           0 0 1 0 0 0             0.1 0.5 1.0 0.5 0.1
  • (x3, y3)           0 0 0 0 0 0             0.0 0.1 0.3 0.1 0.0
                       0 0 0 0 1 0             0.0 0.0 0.1 0.3 0.5
                       0 0 0 0 0 0             0.0 0.0 0.0 0.1 0.3
```

**Properti kunci:** `sum(density_map) ≈ jumlah_objek`

Gaussian filter "menyebarkan" setiap titik menjadi distribusi kontinu. Parameter `sigma` mengontrol lebar sebaran:

| Script | Sigma | Kegunaan |
|--------|-------|----------|
| `generate_density_map.py` | 4 | Visualisasi cepat, objek kecil |
| `generate_ground_truth.py` | 15 | Ground truth training, objek besar |

### Model Architecture

**MobileNetV2** dipilih sebagai backbone karena:
- ✅ Ringan dan efisien (cocok untuk deployment)
- ✅ Pretrained pada ImageNet (transfer learning)
- ✅ Depthwise separable convolution (mengurangi parameter)

**Dilated Convolution** digunakan karena:
- ✅ Receptive field lebih luas tanpa menambah parameter
- ✅ Mempertahankan resolusi spasial
- ✅ Menangkap konteks multi-scale

**Bilinear Upsample** digunakan karena:
- ✅ Mengembalikan resolusi output ke ukuran input
- ✅ Tidak menambah parameter (parameter-free)
- ✅ Menghasilkan transisi piksel yang halus

**Preservasi Resolusi Asli:**
Gambar diproses langsung pada **resolusi aslinya** (tidak di-_resize_ secara kaku ke rasio persegi seperti 224x224). Ini adalah keputusan arsitektur yang krusial untuk mencegah hancurnya detail spasial objek berukuran mikro akibat proses downscaling, memastikan model tetap bisa membedakan titik antar-objek dengan jelas pada density map.

### Format Anotasi (JSON)

```json
{
  "image": "nama_file.png",
  "image_width": 1024,
  "image_height": 1024,
  "count": 7,
  "points": [
    [x1, y1],
    [x2, y2],
    ...
  ]
}
```

> **Penting:** Koordinat menggunakan format `[x, y]` (OpenCV convention). Saat dipetakan ke matriks NumPy, dikonversi menjadi `density[y, x]` karena NumPy menggunakan `(baris, kolom)`.

---

## 💻 Cara Menjalankan

```bash
# 1. Clone / masuk ke folder proyek
cd "CAPSTONE (Density Mapping)"

# 2. Install dependencies
pip install numpy opencv-python scipy matplotlib torch torchvision

# 3. Letakkan gambar ke dataset/images/

# 4. Jalankan Point Labeler untuk anotasi
py point_labeler.py

# 5. Generate ground truth density map
py generate_ground_truth.py

# 6. Test arsitektur model
py model_dme.py

# 7. (Opsional) Demo density map dengan dummy data
py generate_density_map.py

# 8. Training Model (Pastikan sudah generate ground truth terlebih dahulu!)
py train.py

# 9. Prediksi / Inference menggunakan Model terlatih (akan memuat checkpoints/best_dme_model.pth)
py predict.py
```

---

## 🛠 Teknologi yang Digunakan

| Teknologi | Versi | Kegunaan |
|-----------|-------|----------|
| Python | 3.8+ | Bahasa pemrograman utama |
| PyTorch | 2.0+ | Deep learning framework |
| TorchVision | 0.15+ | Pretrained models (MobileNetV2) |
| OpenCV | 4.5+ | Image processing & GUI annotation tool |
| NumPy | 1.21+ | Operasi matriks & penyimpanan density map |
| SciPy | 1.7+ | Gaussian filter untuk density map |
| Matplotlib | 3.4+ | Visualisasi heatmap overlay |

---

## 🗺 Roadmap & Pengembangan

- [x] **Struktur dataset** — Folder `images/`, `annotations/`, `ground_truth/`
- [x] **Point Labeler GUI** — Tool anotasi titik interaktif dengan undo
- [x] **Ground Truth Generator** — Konversi anotasi → density map (.npy + visualisasi)
- [x] **Model Architecture** — MobileNetV2 + Dilated Conv + Upsample
- [x] **Utilitas Density Map** — Fungsi reusable & visualisasi
- [x] **Dataset Loader** — PyTorch `Dataset` dan `DataLoader` dengan augmentasi _Albumentations_ sinkron
- [x] **Training Script** — Loop eksekusi training dengan MSE, Adam, dan _auto-checkpoint_
- [x] **Evaluation Metrics** — Laporan MAE (_Mean Absolute Error_) per-*epoch*
- [x] **Inference Script** — Modul prediksi `predict.py` untuk mengestimasi jumlah objek dengan visualisasi gabungan
- [ ] **Model Export** — Export model ke ONNX untuk deployment
- [ ] **Web Interface** — Dashboard visualisasi hasil prediksi


---

## 📝 Catatan Penting

1. **Koordinat (x, y) vs (y, x):**
   - OpenCV dan JSON menggunakan format `(x, y)` — kolom dulu, baris kemudian
   - NumPy menggunakan format `(y, x)` — baris dulu, kolom kemudian
   - Konversi ini sudah ditangani di semua script

2. **Nilai Sigma:**
   - Sigma kecil (4) → heatmap tajam, cocok untuk objek kecil
   - Sigma besar (15) → heatmap lebar, cocok untuk objek besar dan ground truth

3. **Integritas Jumlah:**
   - `sum(density_map)` harus mendekati jumlah objek yang dianotasi
   - Ini adalah properti fundamental dari Gaussian filter yang mempertahankan integral

---

<p align="center">
  <b>Capstone Project — Density Map Estimation</b><br>
  Built with 🐍 Python | 🔥 PyTorch | 👁 OpenCV
</p>
