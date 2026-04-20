"""
src/inference/predictor.py
Handles:
  - Local model loading (SavedModel format)
  - Single-image prediction
  - Video overlay processing
  - SageMaker endpoint invocation (remote)
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Union

import boto3
import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


#  Local predictor 
class LocalPredictor:
    """
    Runs inference locally using a SavedModel.
    Used during development and batch video processing.
    """

    def __init__(self, model_path: str | Path, img_size: int = 256, threshold: float = 0.5):
        self.img_size  = img_size
        self.threshold = threshold
        logger.info("Loading model from %s", model_path)
        self.model = tf.saved_model.load(str(model_path))
        self._infer = self.model.signatures["serving_default"]

    def predict_image(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: (H, W, 3) uint8 or float32 RGB image
        Returns:
            mask: (H, W) bool road mask
        """
        orig_h, orig_w = img.shape[:2]
        small = cv2.resize(img, (self.img_size, self.img_size))
        small = small.astype(np.float32) / 255.0
        inp   = tf.constant(small[np.newaxis, ...], dtype=tf.float32)

        # SavedModel serving_default may return dict; grab first tensor
        out   = self._infer(inp)
        pred  = list(out.values())[0].numpy()[0, :, :, 0]

        mask  = (pred > self.threshold).astype(np.uint8)
        mask  = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)

    def overlay_mask(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Blend road mask over image.
        Args:
            img:   (H, W, 3) BGR uint8
            mask:  (H, W) bool
            color: BGR overlay color
            alpha: blend factor for mask
        Returns:
            (H, W, 3) BGR overlay image
        """
        overlay = img.copy()
        overlay[mask] = color
        return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        overlay_color: tuple[int, int, int] = (0, 255, 0),
    ) -> Path:
        """
        Read video frame-by-frame, apply road segmentation overlay, write output.
        Returns path to output video.
        """
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask  = self.predict_image(rgb)
            result = self.overlay_mask(frame, mask, color=overlay_color)
            out.write(result)
            frame_idx += 1
            if frame_idx % 30 == 0:
                logger.info("Processed %d/%d frames", frame_idx, total)

        cap.release()
        out.release()
        logger.info("Video saved to %s", output_path)
        return output_path


#  SageMaker endpoint predictor 

class SageMakerPredictor:
    """
    Invokes a deployed SageMaker real-time endpoint.
    Sends image as JPEG bytes; receives JSON mask probabilities.
    """

    def __init__(
        self,
        endpoint_name: str,
        img_size: int = 256,
        threshold: float = 0.5,
        region: str = "us-east-1",
    ):
        self.endpoint_name = endpoint_name
        self.img_size  = img_size
        self.threshold = threshold
        self._runtime  = boto3.client("sagemaker-runtime", region_name=region)

    def predict_image(self, img: np.ndarray) -> np.ndarray:
        orig_h, orig_w = img.shape[:2]
        small = cv2.resize(img, (self.img_size, self.img_size))
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))

        response = self._runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="image/jpeg",
            Body=buf.tobytes(),
        )
        payload = json.loads(response["Body"].read())
        pred = np.array(payload["predictions"], dtype=np.float32)
        mask = (pred > self.threshold).astype(np.uint8)
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)
