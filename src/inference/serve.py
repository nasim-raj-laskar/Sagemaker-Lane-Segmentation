"""
src/inference/serve.py
SageMaker custom inference handler.
SageMaker calls these four functions automatically:

  model_fn      → load model
  input_fn      → deserialize request body
  predict_fn    → run inference
  output_fn     → serialize prediction

Supports content types: image/jpeg, image/png, application/json
"""
from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

_IMG_SIZE  = int(os.environ.get("IMG_SIZE", "256"))
_THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))


def model_fn(model_dir: str):
    """Called once at container startup."""
    saved_model_path = Path(model_dir) / "road_seg_savedmodel"
    if not saved_model_path.exists():
        # fallback: look for any SavedModel-like directory
        saved_model_path = Path(model_dir)
    logger.info("Loading model from %s", saved_model_path)
    model = tf.saved_model.load(str(saved_model_path))
    return model


def input_fn(request_body: bytes, content_type: str) -> np.ndarray:
    """Deserialize incoming request to numpy image array (RGB)."""
    if content_type in ("image/jpeg", "image/png", "image/jpg"):
        buf = np.frombuffer(request_body, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    elif content_type == "application/json":
        payload = json.loads(request_body)
        arr = np.array(payload["image"], dtype=np.float32)
        return arr
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(img: np.ndarray, model) -> np.ndarray:
    """Run forward pass and return probability map."""
    orig_h, orig_w = img.shape[:2]
    small = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE))
    small = small.astype(np.float32) / 255.0
    inp   = tf.constant(small[np.newaxis, ...], dtype=tf.float32)
    infer = model.signatures["serving_default"]
    out   = infer(inp)
    pred  = list(out.values())[0].numpy()[0, :, :, 0]
    # Resize back to original spatial dims
    pred  = cv2.resize(pred, (orig_w, orig_h))
    return pred


def output_fn(prediction: np.ndarray, accept: str) -> tuple[bytes, str]:
    """Serialize prediction for the response."""
    mask = (prediction > _THRESHOLD).astype(int)
    payload = {
        "predictions": mask.tolist(),
        "probabilities": prediction.tolist(),
    }
    return json.dumps(payload).encode(), "application/json"
