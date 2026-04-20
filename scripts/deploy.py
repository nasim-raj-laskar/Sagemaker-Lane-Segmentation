"""
scripts/deploy.py
Deploys an approved model from the SageMaker Model Registry to a real-time endpoint.

Usage:
  python scripts/deploy.py --action deploy
  python scripts/deploy.py --action update --model-version 3
  python scripts/deploy.py --action delete
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import get_config
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


def get_latest_approved_model_uri(group_name: str, region: str) -> str:
    sm = boto3.client("sagemaker", region_name=region)
    packages = sm.list_model_packages(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["ModelPackageSummaryList"]
    if not packages:
        raise RuntimeError(f"No approved models in group '{group_name}'")
    pkg_arn = packages[0]["ModelPackageArn"]
    details = sm.describe_model_package(ModelPackageName=pkg_arn)
    return details["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]


def deploy(cfg=None) -> str:
    cfg = cfg or get_config()
    aws_cfg = cfg.aws

    session  = sagemaker.Session(boto_session=boto3.Session(region_name=aws_cfg.region))
    role_arn = aws_cfg.resolved_role_arn()

    model_uri = get_latest_approved_model_uri(
        group_name="RoadSegmentationModels",
        region=aws_cfg.region,
    )
    logger.info("Deploying model: %s", model_uri)

    model = TensorFlowModel(
        model_data=model_uri,
        role=role_arn,
        framework_version="2.13",
        sagemaker_session=session,
        entry_point="src/inference/serve.py",
        source_dir=".",
        env={
            "IMG_SIZE":  str(cfg.data.img_size),
            "THRESHOLD": str(cfg.inference.threshold),
        },
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=aws_cfg.sagemaker.instance_type["inference"],
        endpoint_name=cfg.inference.endpoint_name,
    )
    logger.info("Endpoint deployed: %s", cfg.inference.endpoint_name)
    return cfg.inference.endpoint_name


def delete_endpoint(cfg=None) -> None:
    cfg = cfg or get_config()
    sm = boto3.client("sagemaker", region_name=cfg.aws.region)
    sm.delete_endpoint(EndpointName=cfg.inference.endpoint_name)
    logger.info("Deleted endpoint: %s", cfg.inference.endpoint_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--action", choices=["deploy", "update", "delete"], required=True)
    p.add_argument("--model-version", type=int, default=None)
    args = p.parse_args()

    cfg = get_config()

    if args.action in ("deploy", "update"):
        endpoint_name = deploy(cfg)
        logger.info("Done. Endpoint: %s", endpoint_name)
    elif args.action == "delete":
        delete_endpoint(cfg)


if __name__ == "__main__":
    main()
