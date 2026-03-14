"""src/utils/s3.py — S3 helpers for upload, download, listing, and model packaging."""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import boto3
from omegaconf import DictConfig

from src.utils.logging import get_logger

logger = get_logger(__name__)


class S3Client:
    """Thin wrapper around boto3 S3 with project-aware helpers."""

    def __init__(self, cfg: DictConfig):
        self.bucket = cfg.data.s3.bucket
        self.region = cfg.aws.region
        self._client = boto3.client("s3", region_name=self.region)

    def upload_file(self, local_path: str | Path, s3_key: str) -> str:
        local_path = str(local_path)
        logger.info("Uploading %s → s3://%s/%s", local_path, self.bucket, s3_key)
        self._client.upload_file(local_path, self.bucket, s3_key)
        return f"s3://{self.bucket}/{s3_key}"

    def upload_dir(self, local_dir: str | Path, s3_prefix: str) -> list[str]:
        """Recursively upload a directory. Returns list of S3 URIs."""
        local_dir = Path(local_dir)
        uris = []
        for f in local_dir.rglob("*"):
            if f.is_file():
                key = s3_prefix.rstrip("/") + "/" + str(f.relative_to(local_dir))
                uris.append(self.upload_file(f, key))
        return uris

    def download_file(self, s3_key: str, local_path: str | Path) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading s3://%s/%s → %s", self.bucket, s3_key, local_path)
        self._client.download_file(self.bucket, s3_key, str(local_path))
        return local_path

    def read_bytes(self, bucket: str, key: str) -> bytes:
        response = self._client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def presigned_url(self, s3_key: str, expiry_seconds: int = 3600) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": s3_key},
            ExpiresIn=expiry_seconds,
        )


def package_model_for_sagemaker(model_dir: str | Path, output_path: str | Path) -> Path:
    """Tar-gz a SavedModel/Keras directory into model.tar.gz for SageMaker."""
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(output_path), "w:gz") as tar:
        tar.add(str(model_dir), arcname=".")
    logger.info("Packaged model → %s  (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def upload_model_artifact(
    model_dir: str | Path,
    cfg: DictConfig,
    version: str = "latest",
) -> str:
    """Package model and upload to S3. Returns the S3 URI."""
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "model.tar.gz"
        package_model_for_sagemaker(model_dir, tar_path)
        s3 = S3Client(cfg)
        key = f"{cfg.aws.s3.model_prefix}/{version}/model.tar.gz"
        return s3.upload_file(tar_path, key)
