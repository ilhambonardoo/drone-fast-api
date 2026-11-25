import os
import json
import cv2
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from pathlib import Path
from uuid import uuid4
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "DreamPalm_Config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    MODEL_CONFIG = json.load(f)

MEAN = np.asarray(MODEL_CONFIG["mean"], dtype=np.float32)
STD = np.asarray(MODEL_CONFIG["std"], dtype=np.float32)

CLIP_MIN = float(MODEL_CONFIG["clip_min"])
CLIP_MAX = float(MODEL_CONFIG["clip_max"])
INVALID_FILL = float(MODEL_CONFIG["invalid_pixel_fill"])
THRESHOLD = float(MODEL_CONFIG["threshold"])
CLASS_MAPPING = MODEL_CONFIG["class_mapping"]

INPUT_H, INPUT_W, INPUT_C = MODEL_CONFIG["input_shape"]
MASK_CFG = MODEL_CONFIG["mask"]
yy, xx = np.ogrid[:INPUT_H, :INPUT_W]
dist_sq = (
    (xx - MASK_CFG["center_x"]) ** 2
    + (yy - MASK_CFG["center_y"]) ** 2
)

CNN_KEEP_MASK = dist_sq > (MASK_CFG["radius_pixels"] ** 2)
CNN_ZERO_MASK = ~CNN_KEEP_MASK

app = FastAPI(
    title="DreamPalm MAPIR CNN & NDVI Processing API",
    version="3.1-local-storage"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hanya penyimpanan hasil analisis. JPG/RAW asli disimpan Next.js.
STORAGE_ROOT = Path(os.getenv(
    "DREAMPALM_STORAGE_ROOT", "/srv/dreampalm-storage/uji"
)).resolve()
MEDIA_BASE = os.getenv("DREAMPALM_MEDIA_BASE", "/api/media").rstrip("/")

KERAS_MODEL_PATH = os.path.join(
    BASE_DIR,
    "DreamPalm_CNN_720_AreaLuar.keras"
)

try:
    cnn_classifier = tf.keras.models.load_model(
        KERAS_MODEL_PATH,
        compile=False
    )
    print(
        f"Model {MODEL_CONFIG['model_name']} berhasil dimuat "
        f"dari {KERAS_MODEL_PATH}"
    )
except Exception as e:
    print(f"Gagal memuat model: {e}")
    cnn_classifier = None

def save_analysis_images(img_720, ndvi_colormap):
    prediction_id = str(uuid4())
    relative_dir = Path("predictions") / prediction_id
    parent = STORAGE_ROOT / "predictions"
    parent.mkdir(parents=True, exist_ok=True)
    temp_dir = parent / f".{prediction_id}.tmp"
    final_dir = STORAGE_ROOT / relative_dir
    temp_dir.mkdir(mode=0o750)
    try:
        for name, ext, image, params in (
            ("processed.jpg", ".jpg", img_720, [cv2.IMWRITE_JPEG_QUALITY, 85]),
            ("ndvi.png", ".png", ndvi_colormap, []),
        ):
            ok, encoded = cv2.imencode(ext, image, params)
            if not ok:
                raise ValueError(f"Gagal melakukan enkoding {name}.")
            (temp_dir / name).write_bytes(encoded.tobytes())
        temp_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    processed_path = (relative_dir / "processed.jpg").as_posix()
    ndvi_path = (relative_dir / "ndvi.png").as_posix()
    return prediction_id, processed_path, ndvi_path


def prepare_mapir_patch(img_bgr: np.ndarray) -> np.ndarray:
    h, w, _ = img_bgr.shape
    size = min(h, w)
    start_x = (w - size) // 2
    start_y = (h - size) // 2

    cropped = img_bgr[
        start_y:start_y + size,
        start_x:start_x + size,
    ]

    return cv2.resize(
        cropped,
        (INPUT_W, INPUT_H),
        interpolation=cv2.INTER_AREA,
    )


def extract_mapir_grn(img_720_bgr: np.ndarray):
    b_nir, g_green, r_red = cv2.split(img_720_bgr)

    return (
        g_green.astype(np.float32),
        r_red.astype(np.float32),
        b_nir.astype(np.float32),
    )

def mapir_jpg_to_model_scale(
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
):
    scale = np.float32(255.0)

    return (
        green / scale,
        red / scale,
        nir / scale,
    )

def calculate_ndvi_and_spectral(
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
):
    stacked = np.stack([green, red, nir], axis=-1).astype(np.float32)

    valid = np.all(np.isfinite(stacked), axis=-1)

    denominator = nir + red

    valid_ndvi = (
        valid
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-8)
    )

    ndvi = np.full(red.shape, np.nan, dtype=np.float32)

    np.divide(
        nir - red,
        denominator,
        out=ndvi,
        where=valid_ndvi,
    )

    avg_ndvi = (
        float(ndvi[valid_ndvi].mean())
        if valid_ndvi.any()
        else None
    )

    if valid.any():
        bands = {
            "green": float(green[valid].mean()),
            "red": float(red[valid].mean()),
            "nir": float(nir[valid].mean()),
        }
    else:
        bands = {"green": None, "red": None, "nir": None}

    return ndvi, avg_ndvi, bands, valid

def preprocess_for_cnn(
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    stacked = np.stack([green, red, nir], axis=-1).astype(np.float32)

    stacked[~valid] = INVALID_FILL

    normalized = np.clip(
        (stacked - MEAN) / STD,
        CLIP_MIN,
        CLIP_MAX,
    )

    normalized[~valid] = INVALID_FILL
    normalized[CNN_ZERO_MASK] = 0.0

    return np.expand_dims(normalized.astype(np.float32), axis=0)

def create_ndvi_heatmap(ndvi: np.ndarray) -> np.ndarray:
    safe_ndvi = np.nan_to_num(
        ndvi,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    safe_ndvi = np.clip(safe_ndvi, -1.0, 1.0)

    ndvi_norm = (
        (safe_ndvi + 1.0) / 2.0 * 255.0
    ).astype(np.uint8)

    return cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_JET)

def process_drone_image(image_bytes: bytes, filename: str):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("File gambar tidak valid.")
    img_720 = prepare_mapir_patch(img)
    green_8, red_8, nir_8 = extract_mapir_grn(img_720)
    green, red, nir = mapir_jpg_to_model_scale(
        green_8,
        red_8,
        nir_8,
    )
    ndvi_matrix, avg_ndvi, bands, valid = calculate_ndvi_and_spectral(
        green,
        red,
        nir,
    )

    if not valid.any():
        raise ValueError("Tidak ditemukan piksel valid.")

    if cnn_classifier is None:
        raise RuntimeError("Model CNN belum berhasil dimuat.")

    tensor = preprocess_for_cnn(
        green,
        red,
        nir,
        valid,
    )

    prediction = cnn_classifier.predict(tensor, verbose=0)

    score_terserang = float(np.squeeze(prediction))

    if not np.isfinite(score_terserang):
        raise ValueError("Output model CNN tidak valid.")

    terserang = score_terserang >= THRESHOLD
    class_idx = int(terserang)

    classification_label = CLASS_MAPPING[str(class_idx)]

    score_sehat = 1.0 - score_terserang
    prediction_score = (
        score_terserang if terserang else score_sehat
    )

    ndvi_colormap = create_ndvi_heatmap(ndvi_matrix)
    prediction_id, processed_path, ndvi_path = save_analysis_images(
        img_720, ndvi_colormap
    )
    url_thumb = f"{MEDIA_BASE}/{processed_path}"
    url_ndvi = f"{MEDIA_BASE}/{ndvi_path}"

    return {
        "success": True,
        "predictionId": prediction_id,
        "processedJpgPath": processed_path,
        "ndviVisualPath": ndvi_path,
        "filename": filename,
        "modelName": MODEL_CONFIG["model_name"],
        "modelStatus": MODEL_CONFIG["status"],
        "classification": classification_label,
        "classIndex": class_idx,
        "scoreTerserang": round(score_terserang, 6),
        "scoreSehat": round(score_sehat, 6),
        "predictionScore": round(prediction_score, 6),
        "scoreCalibrated": False,
        "threshold": THRESHOLD,
        "ndvi": (
            round(avg_ndvi, 6)
            if avg_ndvi is not None
            else None
        ),
        "bands": {
            "green": (
                round(bands["green"], 6)
                if bands["green"] is not None
                else None
            ),
            "red": (
                round(bands["red"], 6)
                if bands["red"] is not None
                else None
            ),
            "nir": (
                round(bands["nir"], 6)
                if bands["nir"] is not None
                else None
            ),
        },
        "validPixelPercentage": round(
            float(valid.mean() * 100.0),
            4,
        ),
        "inputShape": [INPUT_H, INPUT_W, INPUT_C],
        "snapshotThumbnail": url_thumb,
        "snapshotNDVI": url_ndvi,
        "radiometricMode": "mapir_jpg_div255_provisional",
        "radiometricValidated": False,
        "physicalCoverageTargetMeters": 10.0,
        "physicalCoverageValidated": False,
        "automaticSprayingAllowed": MODEL_CONFIG[
            "automatic_spraying_allowed"
        ],
    }

@app.post("/process-snapshot")
async def process_snapshot_endpoint(
    file: UploadFile = File(...),
):
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="File harus berupa gambar.",
        )

    try:
        contents = await file.read()

        return process_drone_image(
            contents,
            file.filename or "capture.jpg",
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
