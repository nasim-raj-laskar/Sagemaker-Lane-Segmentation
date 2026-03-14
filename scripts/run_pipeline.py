"""scripts/run_pipeline.py — build, upsert, and start the SageMaker pipeline.

Usage:
    python scripts/run_pipeline.py                      # run with default config
    python scripts/run_pipeline.py training.epochs=10   # override epochs
    python scripts/run_pipeline.py --dry-run            # build + print definition only
"""

from __future__ import annotations

import argparse
import json
import sys

import hydra
from omegaconf import DictConfig

from src.pipelines.sagemaker_pipeline import build_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dry-run", action="store_true")
    args, _ = p.parse_known_args()

    logger.info("Building SageMaker pipeline '%s'", cfg.aws.pipeline.name)
    pipeline = build_pipeline(cfg)

    if args.dry_run:
        defn = json.loads(pipeline.definition())
        print(json.dumps(defn, indent=2))
        logger.info("Dry run complete — pipeline not started")
        return

    logger.info("Upserting pipeline definition...")
    pipeline.upsert(role_arn=cfg.aws.iam.execution_role_arn)

    logger.info("Starting pipeline execution...")
    execution = pipeline.start(
        parameters={
            "Epochs": cfg.training.epochs,
            "TrainingInstanceType": cfg.training.sagemaker.instance_type,
        }
    )
    logger.info("Execution ARN: %s", execution.arn)
    logger.info(
        "Track at: https://%s.console.aws.amazon.com/sagemaker/home?region=%s#/pipelines/%s/executions/%s",
        cfg.aws.region, cfg.aws.region, cfg.aws.pipeline.name, execution.arn.split("/")[-1],
    )


if __name__ == "__main__":
    main()
