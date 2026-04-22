"""
tests/integration/test_s3_io.py
Integration tests for S3 helpers using moto (no real AWS calls).
"""
from __future__ import annotations

import io
from pathlib import Path

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_s3

from src.data import s3_io

BUCKET = "test-perceptron-bucket"
REGION = "us-east-1"


@pytest.fixture(autouse=True)
def reset_s3_client():
    """Force s3_io to re-create its boto3 client inside each mock context."""
    s3_io._s3 = None
    yield
    s3_io._s3 = None


@mock_s3
class TestS3IO:
    @pytest.fixture()
    def bucket(self):
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        return BUCKET

    def test_list_keys_empty(self, bucket):
        keys = s3_io.list_keys(bucket, "nonexistent/")
        assert keys == []

    def test_upload_and_list(self, bucket, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        s3_io.upload_file(f, bucket, "test/hello.txt")
        keys = s3_io.list_keys(bucket, "test/")
        assert "test/hello.txt" in keys

    def test_download_file(self, bucket, tmp_path):
        # Upload first
        client = boto3.client("s3", region_name=REGION)
        client.put_object(Bucket=bucket, Key="data/file.txt", Body=b"test content")
        dest = tmp_path / "downloaded.txt"
        s3_io.download_file(bucket, "data/file.txt", dest)
        assert dest.read_text() == "test content"

    def test_upload_dir(self, bucket, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub" / "b.txt").write_text("b")
        uris = s3_io.upload_dir(tmp_path, bucket, "prefix")
        assert len(uris) == 2
        keys = s3_io.list_keys(bucket, "prefix/")
        assert any("a.txt" in k for k in keys)
        assert any("b.txt" in k for k in keys)

    def test_read_image_from_s3(self, bucket):
        # Create a small synthetic image and encode as JPEG
        img_rgb = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, buf  = cv2.imencode(".jpg", img_bgr)
        client  = boto3.client("s3", region_name=REGION)
        client.put_object(Bucket=bucket, Key="img/test.jpg", Body=buf.tobytes())

        result = s3_io.read_image_from_s3(bucket, "img/test.jpg")
        assert result.shape == (50, 50, 3)
        assert result.dtype == np.uint8

    def test_key_exists(self, bucket):
        client = boto3.client("s3", region_name=REGION)
        client.put_object(Bucket=bucket, Key="exists.txt", Body=b"x")
        assert s3_io.key_exists(bucket, "exists.txt")
        assert not s3_io.key_exists(bucket, "missing.txt")
