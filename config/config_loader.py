"""
config/config_loader.py
Loads and validates config.yaml; exposes a typed Config object.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
import yaml
from pydantic import BaseModel, field_validator


#Pydantic models (typed, validated) 

class S3Prefix(BaseModel):
    raw_images: str
    raw_masks:  str
    processed:  str
    artifacts:  str
    testing:    str
    pipeline:   str


class SageMakerConfig(BaseModel):
    role_name:     str
    studio_domain_id: str | None = None
    instance_type: dict[str, str]
    ecr: dict[str, str] = {}


class AWSConfig(BaseModel):
    region:     str
    account_id: str | None = None
    s3_bucket:  str
    s3_prefix:  S3Prefix
    sagemaker:  SageMakerConfig
    ecr:        dict[str, str]
    mlflow:     dict[str, Any]

    def resolved_role_arn(self) -> str:
        account = self.account_id or self._get_account_id()
        return f"arn:aws:iam::{account}:role/{self.sagemaker.role_name}"

    @staticmethod
    def _get_account_id() -> str:
        return boto3.client("sts").get_caller_identity()["Account"]

    def s3_uri(self, prefix_key: str, *parts: str) -> str:
        base = getattr(self.s3_prefix, prefix_key)
        path = "/".join([p.strip("/") for p in [base, *parts] if p])
        return f"s3://{self.s3_bucket}/{path}"


class DataConfig(BaseModel):
    img_size:     int
    channels:     int
    num_classes:  int
    val_split:    float
    random_seed:  int
    augmentation: dict[str, Any]


class ModelConfig(BaseModel):
    architecture:     str
    backbone:         str
    backbone_weights: str
    freeze_encoder:   bool
    dropout_rate:     float
    decoder_filters:  list[int]


class TrainingConfig(BaseModel):
    epochs:          int
    batch_size:      int
    learning_rate:   float
    optimizer:       str
    loss:            str
    metrics:         list[str]
    early_stopping:  dict[str, Any]
    reduce_lr:       dict[str, Any]
    checkpoint:      dict[str, Any]


class InferenceConfig(BaseModel):
    threshold:     float
    overlay_color: list[int]
    endpoint_name: str


class Config(BaseModel):
    project:   dict[str, Any]
    aws:       AWSConfig
    data:      DataConfig
    model:     ModelConfig
    training:  TrainingConfig
    inference: InferenceConfig
    monitoring: dict[str, Any]
    pipeline:  dict[str, Any]


#  Loader 

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml and return a validated Config object (cached)."""
    cfg_path = Path(path) if path else _CONFIG_PATH
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def get_config() -> Config:
    """Convenience alias used throughout the codebase."""
    return load_config()
