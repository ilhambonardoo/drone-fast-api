# DreamPalm — FastAPI CNN & NDVI

Layanan pengolahan citra JPG kamera multispektral MAPIR Survey3W RGN untuk klasifikasi indikasi kesehatan tajuk kelapa sawit menggunakan CNN, perhitungan NDVI, dan penyimpanan hasil analisis. Repository ini mempertahankan **dua adapter yang dapat dijalankan secara terpisah**, dengan satu model dan satu konfigurasi ML bersama.

## 1. Pilihan penyimpanan

| Aspek              | Filesystem VPS / lokal                  | Supabase Storage                                     |
| ------------------ | --------------------------------------- | ---------------------------------------------------- |
| File Python        | `main.py`                               | `main_supabase.py`                                   |
| Versi API          | `3.1-local-storage`                     | `3.0`                                                |
| Modul Uvicorn      | `main:app`                              | `main_supabase:app`                                  |
| Mekanisme          | Menulis file langsung ke disk           | Mengunggah melalui SDK Supabase                      |
| Identitas file     | UUID setiap prediksi                    | Nama input yang disanitasi                           |
| Hasil JPG          | `predictions/<uuid>/processed.jpg`      | `<nama>_thumb.jpg`                                   |
| Hasil PNG          | `predictions/<uuid>/ndvi.png`           | `<nama>_ndvi.png`                                    |
| URL gambar         | Relatif, membutuhkan Media API Next.js  | URL publik Supabase (bucket harus mengizinkan akses) |
| Kredensial storage | Tidak diperlukan; izin filesystem Linux | URL, key, dan bucket                                 |

Pemilihan mode dilakukan melalui nama modul Uvicorn, bukan melalui variabel `STORAGE_MODE`. Tidak ada perpindahan otomatis atau upload ke kedua storage sekaligus. Kedua adapter mempertahankan kontrak masing-masing; perubahan mode perlu disesuaikan dengan aplikasi pemanggil.

Mode Supabase dipertahankan sesuai adapter existing, termasuk `upsert=true`: nama foto yang sama dapat menimpa objek sebelumnya. Upload dilakukan sebelum inferensi CNN, sehingga kegagalan inferensi dapat meninggalkan objek hasil pengolahan. Mode VPS menyimpan setelah inferensi berhasil dan menggunakan direktori sementara serta atomic rename; ini mengurangi risiko file parsial, tetapi tidak menjamin ketahanan terhadap semua kegagalan disk atau proses.

## 2. Struktur repository

```text
drone-fast-api/
├── main.py
├── main_supabase.py
├── DreamPalm_CNN_720_AreaLuar.keras
├── DreamPalm_Config.json
├── env.example
├── requirements.txt
├── requirements-vps.txt
├── .gitignore
├── README.md
├── deploy/
│   └── dreampalm-uji-ml-api.service
└── tests/
    └── test_pipeline_consistency.py
```

File `.env`, virtual environment (`.venv/` atau `venv/`), dan folder `storage/` dibuat secara lokal dan tidak dimasukkan ke Git. Model dan konfigurasi JSON harus berada pada direktori yang sama dengan kedua adapter.

## 3. Instalasi lokal

Gunakan versi Python yang kompatibel dengan TensorFlow. Deployment VPS terakhir menggunakan Python 3.12.3 dan TensorFlow 2.21.0. Dependensi menggunakan batas versi minimum, bukan lockfile; versi terinstal pada komputer lain dapat berbeda.

### Windows (PowerShell)

Dari root repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Jika hanya ingin menggunakan VPS/local storage, gunakan `requirements-vps.txt` sebagai pengganti `requirements.txt`. Tidak perlu membuat virtual environment baru jika sudah memiliki environment yang sesuai; gunakan interpreter environment tersebut.

### Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` memuat dependensi dasar melalui `-r requirements-vps.txt` dan menambahkan SDK Supabase. Kedua pilihan instalasi mempertahankan `python-dotenv` untuk membaca `.env`.

## 4. Konfigurasi environment

Salin `env.example` menjadi `.env` jika belum tersedia. Jangan menimpa `.env` lama yang berisi kredensial.

```powershell
if (-not (Test-Path .env)) { Copy-Item env.example .env }
```

Pada Linux, dapat menggunakan `test -e .env || cp env.example .env`.

| Variabel                 | Mode        | Fungsi                                                                 |
| ------------------------ | ----------- | ---------------------------------------------------------------------- |
| `DREAMPALM_STORAGE_ROOT` | VPS / lokal | Direktori fisik hasil analisis; path relatif mengikuti direktori kerja |
| `DREAMPALM_MEDIA_BASE`   | VPS / lokal | Prefix URL media dalam response, bukan alamat server storage           |
| `SUPABASE_URL`           | Supabase    | Alamat project Supabase                                                |
| `SUPABASE_KEY`           | Supabase    | Key backend yang memiliki izin upload                                  |
| `SUPABASE_BUCKET`        | Supabase    | Nama bucket, contoh `snapshots`                                        |

Contoh `.env` untuk filesystem lokal:

```env
DREAMPALM_STORAGE_ROOT=./storage
DREAMPALM_MEDIA_BASE=/api/media
```

Contoh `.env` untuk Supabase:

```env
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_BUCKET=snapshots
```

Isi URL dan key Supabase secara pribadi sebelum menjalankan mode tersebut. Bucket harus sudah ada. Jangan membiarkan `SUPABASE_BUCKET` kosong karena adapter hanya memakai nilai default jika variabel tidak disetel. Jangan commit key, termasuk service role key yang memiliki hak istimewa tinggi; gunakan kredensial backend dengan izin yang diperlukan. URL hasil adapter lama menggunakan `get_public_url`, sehingga bucket privat tidak otomatis dapat diakses melalui URL tersebut.

Mode VPS tidak membutuhkan key Supabase maupun `DATABASE_URL`. Bila kedua kelompok variabel ada dalam `.env`, masing-masing adapter hanya menggunakan kelompok yang relevan. Pada VPS uji, systemd memberikan konfigurasi langsung; `.env` khusus tidak diperlukan.

## 5. Menjalankan API

Pilih salah satu perintah berikut dari root repository (contoh Windows PowerShell):

**Filesystem lokal / VPS adapter:**

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8002 --workers 1
```

**Supabase adapter:**

```powershell
.\.venv\Scripts\python.exe -m uvicorn main_supabase:app --host 127.0.0.1 --port 8001 --workers 1
```

Di Linux, gunakan `.venv/bin/python` sebagai interpreter. Port merupakan pilihan saat menjalankan Uvicorn, bukan bagian dari kode adapter. Jika kedua mode dijalankan sekaligus, gunakan port berbeda dan perhatikan penggunaan RAM/CPU karena masing-masing memuat model sendiri.

Swagger UI: `http://127.0.0.1:8002/docs` atau `http://127.0.0.1:8001/docs` sesuai mode. `127.0.0.1` mengacu pada komputer yang menjalankan API. Next.js di laptop tidak dapat mengakses localhost VPS secara langsung tanpa tunnel atau deployment pada VPS yang sama.

## 6. Model dan pipeline pengolahan

Kedua adapter menggunakan model dan konfigurasi yang sama:

| Parameter                                      | Nilai                                                   |
| ---------------------------------------------- | ------------------------------------------------------- |
| Model                                          | `DreamPalm_CNN_720_AreaLuar.keras`                      |
| Konfigurasi                                    | `DreamPalm_Config.json`                                 |
| Status                                         | `prototype_exploratory`                                 |
| Kamera data training                           | DJI Mavic 3 Multispectral                               |
| Sampel training/eksperimen menurut konfigurasi | 40                                                      |
| Input                                          | 720 × 720 × 3                                           |
| Kanal CNN                                      | Green, Red, NIR                                         |
| Normalisasi                                    | Z-score, clipping [-5, 5]                               |
| Mask                                           | Pusat (359,5; 359,5), radius 220 piksel; pusat dinolkan |
| Threshold                                      | 0,5                                                     |
| Kelas 0 / 1                                    | Sehat / Terserang                                       |

Urutan pengolahan:

1. Dekode gambar menggunakan OpenCV, crop persegi dari tengah, dan resize 720 × 720.
2. Interpretasikan BGR sebagai NIR, Green, Red; susun kanal GRN dan bagi nilai kanal dengan 255 (provisional).
3. Hitung NDVI = (NIR − Red) / (NIR + Red) dan rata-rata kanal sebelum Z-score dan masking CNN. NDVI rata-rata hanya memakai piksel dengan penyebut valid; jika tidak ada, hasil `null`.
4. Isi piksel tidak valid sesuai konfigurasi, lakukan Z-score dan clipping, lalu nolkan area tengah mask untuk input CNN.
5. Jalankan inferensi sigmoid; skor kelas terserang dibandingkan dengan threshold, skor sehat = 1 − skor terserang.
6. Buat visual NDVI menggunakan colormap JET. Adapter VPS menyimpan kedua gambar setelah inferensi; adapter Supabase existing mengunggah gambar sebelum inferensi.

`processed.jpg` atau thumbnail adalah hasil crop/resize citra, bukan foto asli atau tensor CNN yang sudah dinormalisasi. Nilai `bands` adalah rata-rata kanal berskala /255, bukan reflektansi terkalibrasi. `validPixelPercentage` mengukur piksel dengan kanal berhingga, bukan persentase tajuk sehat atau validitas NDVI.

## 7. Kontrak endpoint

Kedua mode menggunakan `POST /process-snapshot` dengan `multipart/form-data`, field `file`. Input yang ditargetkan adalah JPG MAPIR RGN. Kode memeriksa tipe `image/*` dan keberhasilan decode OpenCV; menerima format gambar lain tidak berarti format tersebut sudah tervalidasi secara ilmiah.

Contoh pengujian Windows (gunakan file gambar yang tersedia):

```powershell
curl.exe -X POST http://127.0.0.1:8002/process-snapshot -F "file=@test.jpg;type=image/jpeg"
```

Untuk Supabase, ganti port sesuai proses yang berjalan. Contoh response VPS berikut berasal dari uji sintetis; UUID dan URL hanya contoh, bukan media aktif.

```json
{
  "success": true,
  "predictionId": "8ea1ad6d-788e-41a9-b76b-06d0481a58fa",
  "processedJpgPath": "predictions/8ea1ad6d-788e-41a9-b76b-06d0481a58fa/processed.jpg",
  "ndviVisualPath": "predictions/8ea1ad6d-788e-41a9-b76b-06d0481a58fa/ndvi.png",
  "filename": "test.jpg",
  "modelName": "DreamPalm_CNN_720_AreaLuar",
  "modelStatus": "prototype_exploratory",
  "classification": "Sehat",
  "classIndex": 0,
  "scoreTerserang": 0.000766,
  "scoreSehat": 0.999234,
  "predictionScore": 0.999234,
  "scoreCalibrated": false,
  "threshold": 0.5,
  "ndvi": 0.60177,
  "bands": { "green": 0.352941, "red": 0.176471, "nir": 0.709804 },
  "validPixelPercentage": 100.0,
  "inputShape": [720, 720, 3],
  "snapshotThumbnail": "/api/media/predictions/8ea1ad6d-788e-41a9-b76b-06d0481a58fa/processed.jpg",
  "snapshotNDVI": "/api/media/predictions/8ea1ad6d-788e-41a9-b76b-06d0481a58fa/ndvi.png",
  "radiometricMode": "mapir_jpg_div255_provisional",
  "radiometricValidated": false,
  "physicalCoverageTargetMeters": 10.0,
  "physicalCoverageValidated": false,
  "automaticSprayingAllowed": false
}
```

Perbedaan kontrak Supabase:

- Tidak mengembalikan `predictionId`, `processedJpgPath`, dan `ndviVisualPath`.
- `snapshotThumbnail` dan `snapshotNDVI` berisi URL Supabase, misalnya `https://example.supabase.co/storage/v1/object/public/snapshots/test_thumb.jpg` dan `.../test_ndvi.png`.
- Field analisis lainnya tetap tersedia dengan struktur yang sama.

Error: HTTP 400 untuk gambar tidak valid, HTTP 422 untuk field request yang tidak memenuhi kontrak FastAPI, atau HTTP 500 untuk kegagalan internal seperti model atau storage. Kedua adapter existing mengembalikan detail exception; gunakan hanya pada jaringan internal tepercaya dan jangan mengekspos respons internal secara langsung ke pengguna publik.

Contoh pemanggilan dari backend Next.js (server-side):

```typescript
const formData = new FormData();
formData.append("file", imageBlob, `${captureId}.jpg`);
const mlRes = await fetch(`${process.env.FASTAPI_URL}/process-snapshot`, {
  method: "POST",
  body: formData,
});
if (!mlRes.ok) throw new Error("Analisis ML gagal");
const mlResult = await mlRes.json();
if (!mlResult.success) throw new Error("Hasil ML tidak valid");
```

Kode contoh mengasumsikan `imageBlob` dan `captureId` sudah tersedia. `FASTAPI_URL` adalah variabel Next.js, bukan konfigurasi adapter Python. Pada deployment web uji nantinya nilainya `http://127.0.0.1:8002`. Backend perlu menangani timeout, validasi respons, dan perbedaan kontrak kedua mode. Schema PostgreSQL uji saat ini dirancang untuk kontrak VPS; mengganti mode ke Supabase tidak otomatis kompatibel dengan pencatatan tersebut.

## 8. Deployment VPS yang sudah ada

| Komponen                    | Konfigurasi terakhir                                |
| --------------------------- | --------------------------------------------------- |
| FastAPI existing            | `/srv/dreampalm-ml-api`, port 8001 (Supabase)       |
| FastAPI web uji             | `/srv/dreampalm-uji-ml-api`, port 8002 (filesystem) |
| Service uji                 | `dreampalm-uji-ml-api.service`                      |
| Host                        | `127.0.0.1`                                         |
| User/group Linux            | `dreampalm`                                         |
| Storage root uji            | `/srv/dreampalm-storage/uji`                        |
| Virtual environment bersama | `/srv/dreampalm-ml-api/.venv`                       |

Contoh konfigurasi systemd sesuai deployment saat ini disimpan di `deploy/dreampalm-uji-ml-api.service`. File dalam repository hanya dokumentasi/template, bukan service yang otomatis aktif. Service yang berjalan berada di `/etc/systemd/system/dreampalm-uji-ml-api.service`.

Systemd menjalankan Uvicorn, memberikan environment, mengaktifkan service saat boot, dan mencoba restart saat gagal. Python menulis gambar langsung ke filesystem menggunakan permission akun Linux. Konfigurasi storage pada unit tersebut:

```ini
Environment="DREAMPALM_STORAGE_ROOT=/srv/dreampalm-storage/uji"
Environment="DREAMPALM_MEDIA_BASE=/api/media"
```

`UMask=0027` membatasi permission file baru. Folder uji telah disiapkan dengan owner `dreampalm:dreampalm` dan mode `750`. Kedua service memakai akun Linux dan virtual environment bersama, sehingga pemisahan proses tidak berarti isolasi keamanan penuh. Jangan uninstall Supabase atau upgrade library pada environment bersama tanpa koordinasi.

Pemeriksaan tanpa mengubah layanan:

```bash
systemctl status dreampalm-uji-ml-api.service --no-pager
systemctl is-enabled dreampalm-uji-ml-api.service
sudo journalctl -u dreampalm-uji-ml-api.service -n 30 --no-pager
curl -s -o /dev/null -w 'HTTP Status: %{http_code}\n' http://127.0.0.1:8002/docs
```

Commit/push GitHub tidak otomatis memperbarui deployment manual. Tidak perlu memasang ulang unit yang sudah aktif untuk perubahan dokumentasi repository ini.

## 9. Struktur storage dan pembagian tugas

```text
/srv/dreampalm-storage/uji/
├── captures/                    # Rencana: Next.js
│   └── <capture_id>/
│       ├── original.jpg
│       └── original.raw
└── predictions/                 # FastAPI VPS
    └── <prediction_id>/
        ├── processed.jpg
        └── ndvi.png
```

FastAPI hanya menghasilkan dua file hasil analisis. FastAPI tidak mengunduh atau menyimpan RAW/JPG asli, tidak menulis metadata ke PostgreSQL, dan tidak menyediakan endpoint Media API. File path relatif pada response VPS harus diselesaikan oleh backend terhadap storage root; URL `/api/media/...` belum otomatis dapat dibuka dari browser.

Raspberry Pi/MAPIR bertanggung jawab pada capture. Next.js nantinya mengunduh JPG/RAW asli, mencatat metadata capture ke `uji.captures`, mengirim JPG ke FastAPI, mencatat hasil ke `uji.predictions`, menyajikan file melalui Media API yang terkontrol, dan menampilkan hasil terbaru. PostgreSQL menyimpan metadata dan relative path, bukan file gambar fisik.

## 10. Pengujian dan status

Pengujian statis repository (tanpa TensorFlow atau Supabase aktif):

```bash
python -m unittest discover -s tests -v
```

Tes ini memeriksa sintaks kedua adapter dan kesamaan AST enam fungsi pengolahan numerik. Tes ini tidak membuktikan seluruh response atau inferensi identik dan tidak menggantikan pengujian integrasi.

Status berdasarkan pengujian VPS pada 25 November 2026:

- [x] Model/config dan adapter VPS sesuai checksum file yang digunakan.
- [x] Inference TensorFlow CPU dan penyimpanan JPG/PNG lokal berhasil pada port 8002.
- [x] Hasil numerik identik untuk satu gambar sintetis pada eksekusi manual dan systemd.
- [x] FastAPI existing port 8001 masih aktif pada pemeriksaan terakhir.
- [x] Schema PostgreSQL uji dan direktori storage telah diuji secara terpisah.
- [ ] Pengujian ulang upload Supabase dengan kredensial aktif untuk paket repository ini.
- [ ] Integrasi Next.js, Media API, dan pencatatan otomatis PostgreSQL.
- [ ] Deployment Next.js uji port 3001 serta autentikasi/Nginx/HTTPS.
- [ ] Pengujian end-to-end menggunakan kamera MAPIR fisik.

## 11. Batasan dan pemeliharaan

Model masih eksploratif dan belum mempunyai validasi independen. Pembagian 255 pada JPG MAPIR bukan kalibrasi reflektansi yang sudah tervalidasi; cakupan fisik target 10 × 10 m belum tervalidasi. Skor CNN bukan probabilitas penyakit terkalibrasi. Pengujian sintetis hanya membuktikan fungsi teknis, bukan akurasi lapangan. Penyemprotan otomatis belum diizinkan.

Kedua adapter sengaja dipertahankan sesuai versi yang sudah digunakan, termasuk duplikasi enam fungsi numerik. Jika pipeline ML kelak berubah, koordinasikan dengan tim ML, perbarui kedua adapter bersama-sama, dan jalankan tes konsistensi serta pengujian inferensi sebelum deployment. Refactoring modul ML bersama dapat dilakukan sebagai PR terpisah.

Jaga API pada localhost, jangan membuka port langsung ke internet, jangan memasukkan kredensial ke Git, dan gunakan autentikasi pada aplikasi pemanggil. Siapkan monitoring kapasitas disk, kebijakan retensi, backup database beserta file, serta prosedur penanganan orphan files sebelum penggunaan rutin. Mode Supabase dan filesystem tidak saling memigrasikan data secara otomatis.
