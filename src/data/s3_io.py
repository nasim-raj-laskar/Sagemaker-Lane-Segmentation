"""
src/data/s3_io.py
All S3 read/write helpers for the project.
Uses smart_open for streaming so we never OOM on large datasets.
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Generator

import boto3
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_s3 = None


def _client() -> boto3.client:
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


#  List / exists 

def list_keys(bucket: str, prefix: str) -> list[str]:
    """Return all object keys under a prefix (handles pagination)."""
    paginator = _client().get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def key_exists(bucket: str, key: str) -> bool:
    try:
        _client().head_object(Bucket=bucket, Key=key)
        return True
    except _client().exceptions.ClientError:
        return False


#  Download 

def download_file(bucket: str, key: str, local_path: str | Path) -> Path:
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Downloading s3://%s/%s → %s", bucket, key, local_path)
    _client().download_file(bucket, key, str(local_path))
    return local_path


def download_prefix(
    bucket: str, prefix: str, local_dir: str | Path
) -> list[Path]:
    """Download all objects under prefix into local_dir, preserving sub-keys."""
    local_dir = Path(local_dir)
    keys = list_keys(bucket, prefix)
    paths = []
    for key in keys:
        rel = key[len(prefix):].lstrip("/")
        dest = local_dir / rel
        paths.append(download_file(bucket, key, dest))
    logger.info("Downloaded %d files from s3://%s/%s", len(paths), bucket, prefix)
    return paths


def read_image_from_s3(bucket: str, key: str) -> np.ndarray:
    """Download image bytes and decode directly into numpy array (RGB)."""
    response = _client().get_object(Bucket=bucket, Key=key)
    buf = np.frombuffer(response["Body"].read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_mask_from_s3(bucket: str, key: str) -> np.ndarray:
    """Return road binary mask (float32, 0/1) from S3 mask key."""
    response = _client().get_object(Bucket=bucket, Key=key)
    buf = np.frombuffer(response["Body"].read(), dtype=np.uint8)
    mask_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    road = (mask_bgr[:, :, 0] == 255).astype(np.float32)
    return road


#  Upload 

def upload_file(local_path: str | Path, bucket: str, key: str) -> str:
    """Upload a local file and return its s3:// URI."""
    logger.debug("Uploading %s → s3://%s/%s", local_path, bucket, key)
    _client().upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def upload_dir(local_dir: str | Path, bucket: str, prefix: str) -> list[str]:
    """Recursively upload a local directory under prefix."""
    local_dir = Path(local_dir)
    uris = []
    for path in local_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(local_dir)
            key = f"{prefix.rstrip('/')}/{rel}"
            uris.append(upload_file(path, bucket, key))
    logger.info("Uploaded %d files to s3://%s/%s", len(uris), bucket, prefix)
    return uris


def upload_numpy(arr: np.ndarray, bucket: str, key: str, ext: str = ".npy") -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    _client().put_object(Bucket=bucket, Key=key, Body=buf.read())
    return f"s3://{bucket}/{key}"
