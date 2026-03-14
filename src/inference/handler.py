"""src/inference/handler.py — SageMaker inference container handler.

SageMaker calls these four functions in order for every request:
    model_fn    → load model once at container start
    input_fn    → deserialise raw request bytes into a tensor
    predict_fn  → run forward pass
    output_fn   → serialise prediction back to bytes

Environment variables injected by SageMaker:
    SM_MODEL_DIR → directory containing the unpacked model.tar.gz
"""

from __future__ import annotations

import base64
import io
import json
import os

import cv2
import numpy as np
import tensorflow as tf

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "artifacts/model")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "256"))
THRESHOLD = float(os.environ.get("MASK_THRESHOLD", "0.5"))


def model_fn(model_dir: str) -> tf.keras.Model:
    """Load the Keras model from model_dir (called once at container start)."""
    model_path = os.path.join(model_dir, "model.keras")
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def input_fn(request_body: bytes, content_type: str) -> np.ndarray:
    """Deserialise raw bytes into a pre-processed image batch (1, H, W, 3)."""
    if content_type in ("image/jpeg", "image/png", "application/octet-stream"):
        img_array = np.frombuffer(request_body, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from request body")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif content_type == "application/json":
        data = json.loads(request_body)
        img_bytes = base64.b64decode(data["image"])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    original_h, original_w = img.shape[:2]
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    # Store original dims in a wrapper so output_fn can upscale
    return {"input": img_resized[np.newaxis, ...], "original_shape": (original_h, original_w)}


def predict_fn(input_data: dict, model: tf.keras.Model) -> dict:
    """Run inference and return raw prediction + original shape."""
    pred = model.predict(input_data["input"], verbose=0)  # (1, H, W, 1)
    return {"prediction": pred[0, :, :, 0], "original_shape": input_data["original_shape"]}


def output_fn(prediction: dict, accept: str) -> tuple[bytes, str]:
    """Serialise the prediction mask to bytes."""
    raw_mask = prediction["prediction"]
    orig_h, orig_w = prediction["original_shape"]

    # Binarise
    binary_mask = (raw_mask > THRESHOLD).astype(np.uint8) * 255

    # Upscale back to original resolution
    mask_full = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if accept in ("image/png", "application/octet-stream"):
        _, buffer = cv2.imencode(".png", mask_full)
        return buffer.tobytes(), "image/png"

    # Default: JSON with base64-encoded PNG + confidence stats
    _, buffer = cv2.imencode(".png", mask_full)
    mask_b64 = base64.b64encode(buffer).decode("utf-8")
    road_pct = float(np.mean(mask_full > 0) * 100)
    response = {
        "mask_png_base64": mask_b64,
        "road_coverage_pct": round(road_pct, 2),
        "mask_shape": list(mask_full.shape),
        "threshold_used": THRESHOLD,
    }
    return json.dumps(response).encode("utf-8"), "application/json"
