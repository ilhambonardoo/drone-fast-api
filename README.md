# DreamPalm MAPIR CNN & NDVI Processing API (v3.0)

Layanan API berbasis **FastAPI** yang berfungsi untuk memproses citra multispektral dari kamera **MAPIR Survey3**, mengekstraksi indeks vegetasi **NDVI** (*Normalized Difference Vegetation Index*) dan kanal spektral (**Green, Red, NIR**), menjalankan inferensi kesehatan tanaman sawit berbasis **Deep Learning CNN (`DreamPalm_CNN_720_AreaLuar.keras`)**, serta mengunggah *thumbnail* preview dan visualisasi *heatmap* NDVI ke **Supabase Storage**.

---

## 🛠️ Fitur Utama

- **Konfigurasi Model Dinamis (`DreamPalm_Config.json`)**: Memuat seluruh parameter preprocessing (*mean*, *std*, *clipping*, dimensi input, *circular mask*, *threshold*, dan pemetaan kelas) secara otomatis dari file konfigurasi JSON.
- **Preprocessing AreaLuar & Z-Score Normalization**:
  - Melakukan *center-crop* persegi dan *resize* ke $720 \times 720$ piksel.
  - Mengekstraksi dan menyusun urutan kanal sesuai standar model: `[Green, Red, NIR]`.
  - Melakukan normalisasi Z-score $\frac{x - \mu}{\sigma}$, *clipping* pada rentang $[-5, 5]$, serta menerapkan *circular mask* dengan menolkan area tengah ($r \le 220\text{ px}$) dan mempertahankan area kanopi luar.
- **Ekstraksi NDVI & Rata-rata Spektral**: Menghitung nilai rata-rata `ndvi`, rata-rata reflektansi per kanal (`green`, `red`, `nir`), serta persentase piksel valid (`validPixelPercentage`) sebelum normalisasi model.
- **Penyimpanan Cloud Hemat Kuota (Supabase Storage)**:
  - Mengunggah **2 file teroptimasi** per capture langsung dari memori (RAM):
    1. `{nama}_thumb.jpg` (JPEG terkompresi kualitas 85%, $\pm 100\text{ KB}$) untuk rendering UI yang cepat.
    2. `{nama}_ndvi.png` (Visualisasi *colormap* JET) untuk analisis indeks vegetasi.
  - Dilengkapi sanitasi nama file otomatis agar aman dari error `InvalidKey`.
- **Metadata & Guardrail Transparansi**: Menyertakan status validasi radiometrik (`radiometricValidated: false`), status kalibrasi skor (`scoreCalibrated: false`), serta pengaman *mode advisory* (`automaticSprayingAllowed: false`).

---

## 📁 Struktur Direktori Proyek

```text
drone/
├── .venv/                           # Virtual Environment Python
├── .env                             # Konfigurasi Kredensial Environment (Jangan di-commit)
├── .gitignore                       # Aturan Git Ignore
├── DreamPalm_Config.json            # Konfigurasi Resmi Preprocessing & Parameter CNN
├── DreamPalm_CNN_720_AreaLuar.keras # Model CNN TensorFlow/Keras (Single-Sigmoid Output)
├── main.py                          # Script Utama FastAPI Service (v3.0)
└── README.md                        # Dokumentasi Proyek
```

---

## ⚙️ Persyaratan Sistem & Dependensi

- **Python**: v3.10 – v3.12
- **Library Python Utama**:
  - `fastapi`
  - `uvicorn`
  - `tensorflow`
  - `opencv-python`
  - `numpy`
  - `supabase`
  - `python-dotenv`
  - `python-multipart`

---

## 🚀 Panduan Instalasi & Pengoperasian

### 1. Navigasi ke Direktori Service
```bash
cd ml-service/drone
```

### 2. Buat & Aktifkan Virtual Environment
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  \.venv\Scripts\Activate.ps1
  ```

### 3. Instal Dependensi
```bash
pip install fastapi uvicorn tensorflow opencv-python numpy supabase python-dotenv python-multipart
```

### 4. Konfigurasi Environment (`.env`)
Buat file `.env` di dalam folder proyek dan isi kredensial Supabase Anda:

```env
SUPABASE_URL=https://xxxxxxxxx.supabase.co
SUPABASE_KEY=your_supabase_service_role_key
SUPABASE_BUCKET=snapshots
```

> 💡 **Catatan Keamanan:** Gunakan **`service_role` key** pada backend FastAPI agar proses unggah gambar ke bucket `snapshots` berjalan tanpa terhalang kebijakan *Row Level Security* (RLS).

---

## 🏃‍♂️ Menjalankan Server API

Jalankan server Uvicorn secara lokal di port `8000`:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- **Base URL:** `http://127.0.0.1:8000`
- **Swagger UI (Dokumentasi Interaktif):** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## 📑 Dokumentasi Endpoint API

### Process Snapshot Image
Menerima file gambar JPG dari kamera MAPIR (melalui proxy Next.js BFF), mengekstraksi nilai NDVI & kanal spektral, menjalankan inferensi CNN sesuai `DreamPalm_Config.json`, dan mengunggah hasil visualisasinya ke Supabase Storage.

- **URL:** `/process-snapshot`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Request Body:**
  - `file`: File gambar (`image/jpeg`, `image/png`)

- **Response Sukses (`200 OK`):**
  ```json
  {
    "success": true,
    "filename": "CAP-20261122-203319-BC28.jpg",
    "modelName": "DreamPalm_CNN_720_AreaLuar",
    "modelStatus": "prototype_exploratory",
    "classification": "Terserang",
    "classIndex": 1,
    "scoreTerserang": 0.877512,
    "scoreSehat": 0.122488,
    "predictionScore": 0.877512,
    "scoreCalibrated": false,
    "threshold": 0.5,
    "ndvi": 0.033418,
    "bands": {
      "green": 0.24152,
      "red": 0.21043,
      "nir": 0.22498
    },
    "validPixelPercentage": 100.0,
    "inputShape": [720, 720, 3],
    "snapshotThumbnail": "https://xxxx.supabase.co/storage/v1/object/public/snapshots/CAP-20261122-203319-BC28_thumb.jpg",
    "snapshotNDVI": "https://xxxx.supabase.co/storage/v1/object/public/snapshots/CAP-20261122-203319-BC28_ndvi.png",
    "radiometricMode": "mapir_jpg_div255_provisional",
    "radiometricValidated": false,
    "physicalCoverageTargetMeters": 10.0,
    "physicalCoverageValidated": false,
    "automaticSprayingAllowed": false
  }
  ```

- **Response Error (`400 Bad Request` / `500 Internal Server Error`):**
  ```json
  {
    "detail": "File harus berupa gambar."
  }
  ```

---

## 🔗 Contoh Integrasi dari Next.js API Route

```typescript
const formData = new FormData();
formData.append("file", imageBlob, `${captureId}.jpg`);

const mlRes = await fetch("http://127.0.0.1:8000/process-snapshot", {
  method: "POST",
  body: formData,
});

const mlResult = await mlRes.json();
console.log(mlResult.classification);    // "Sehat" | "Terserang"
console.log(mlResult.predictionScore);   // Skor prediksi kelas terpilih (0..1)
console.log(mlResult.ndvi);              // Nilai rata-rata NDVI
console.log(mlResult.snapshotThumbnail); // URL Thumbnail JPG ringan
console.log(mlResult.snapshotNDVI);      // URL Visualisasi Heatmap NDVI
```