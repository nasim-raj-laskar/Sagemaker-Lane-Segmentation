"""scripts/deploy.py — deploy or update the SageMaker real-time endpoint.

Usage:
    python scripts/deploy.py                          # deploy with default config
    python scripts/deploy.py aws.endpoint.name=my-ep  # override endpoint name
    python scripts/deploy.py --model-uri s3://...     # deploy a specific model artifact
"""

from __future__ import annotations

import argparse
import sys

import boto3
import hydra
from omegaconf import DictConfig
from sagemaker.tensorflow import TensorFlowModel

from src.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model-uri", default=None, help="Override S3 model URI")
    p.add_argument("--update-only", action="store_true", help="Update existing endpoint config only")
    args, _ = p.parse_known_args()

    import sagemaker
    session = sagemaker.Session(boto_session=boto3.Session(region_name=cfg.aws.region))
    role = cfg.aws.iam.execution_role_arn
    ep_cfg = cfg.aws.endpoint

    model_uri = args.model_uri or (
        f"s3://{cfg.aws.s3.bucket}/{cfg.aws.s3.model_prefix}/latest/model.tar.gz"
    )
    logger.info("Deploying model from %s", model_uri)

    model = TensorFlowModel(
        model_data=model_uri,
        role=role,
        framework_version="2.15",
        entry_point="src/inference/handler.py",
        source_dir=".",
        name=ep_cfg.model_name,
        env={
            "IMG_SIZE": str(cfg.data.preprocessing.img_size),
            "MASK_THRESHOLD": str(cfg.inference.threshold),
        },
        sagemaker_session=session,
    )

    # Check if endpoint already exists
    sm_client = boto3.client("sagemaker", region_name=cfg.aws.region)
    existing = [
        ep["EndpointName"]
        for ep in sm_client.list_endpoints()["Endpoints"]
        if ep["EndpointName"] == ep_cfg.name
    ]

    if existing and args.update_only:
        logger.info("Updating existing endpoint '%s'", ep_cfg.name)
        predictor = model.deploy(
            initial_instance_count=ep_cfg.instance_count,
            instance_type=ep_cfg.instance_type,
            endpoint_name=ep_cfg.name,
            update_endpoint=True,
            wait=True,
        )
    else:
        logger.info("Creating endpoint '%s'", ep_cfg.name)
        predictor = model.deploy(
            initial_instance_count=ep_cfg.instance_count,
            instance_type=ep_cfg.instance_type,
            endpoint_name=ep_cfg.name,
            wait=True,
        )

    logger.info("Endpoint '%s' is live", ep_cfg.name)

    # Smoke test
    import numpy as np
    dummy = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    import cv2
    _, buf = cv2.imencode(".jpg", dummy)
    resp = predictor.predict(buf.tobytes(), initial_args={"ContentType": "image/jpeg"})
    logger.info("Smoke test passed — response keys: %s", list(resp.keys()) if isinstance(resp, dict) else type(resp))


if __name__ == "__main__":
    main()
