"""
scripts/setup_monitoring.py
Sets up SageMaker Model Monitor for the deployed endpoint:
  - Data Quality monitor  (detects input distribution shifts)
  - Model Quality monitor (detects accuracy / IoU degradation)

Run ONCE after deploying the endpoint:
  python scripts/setup_monitoring.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import boto3
import sagemaker
from sagemaker.model_monitor import (
    CronExpressionGenerator,
    DataCaptureConfig,
    DefaultModelMonitor,
    ModelQualityMonitor,
)
from sagemaker.model_monitor.dataset_format import DatasetFormat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import get_config
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


def enable_data_capture(cfg) -> None:
    """(Re-)deploy endpoint with data capture enabled."""
    sm = boto3.client("sagemaker", region_name=cfg.aws.region)
    session = sagemaker.Session(boto_session=boto3.Session(region_name=cfg.aws.region))

    capture_uri = cfg.aws.s3_uri("artifacts", "data-capture")
    capture_cfg = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=20,   # capture 20% of requests
        destination_s3_uri=capture_uri,
        capture_options=["Input", "Output"],
    )

    # Update endpoint config to enable capture
    endpoint_cfg_name = f"{cfg.inference.endpoint_name}-capture-config"
    current = sm.describe_endpoint(EndpointName=cfg.inference.endpoint_name)
    current_cfg = sm.describe_endpoint_config(
        EndpointConfigName=current["EndpointConfigName"]
    )

    sm.create_endpoint_config(
        EndpointConfigName=endpoint_cfg_name,
        ProductionVariants=current_cfg["ProductionVariants"],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 20,
            "DestinationS3Uri": capture_uri,
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
        },
    )
    sm.update_endpoint(
        EndpointName=cfg.inference.endpoint_name,
        EndpointConfigName=endpoint_cfg_name,
    )
    logger.info("Data capture enabled → %s", capture_uri)


def setup_data_quality_monitor(cfg) -> None:
    session  = sagemaker.Session(boto_session=boto3.Session(region_name=cfg.aws.region))
    role_arn = cfg.aws.resolved_role_arn()

    monitor = DefaultModelMonitor(
        role=role_arn,
        instance_count=1,
        instance_type=cfg.aws.sagemaker.instance_type["processing"],
        volume_size_in_gb=20,
        sagemaker_session=session,
    )

    monitor.create_monitoring_schedule(
        endpoint_input=cfg.inference.endpoint_name,
        output_s3_uri=cfg.aws.s3_uri("artifacts", "monitor-reports/data-quality"),
        statistics=sagemaker.model_monitor.Statistics.from_s3_uri(
            cfg.aws.s3_uri("artifacts", "monitor-baseline/statistics.json")
        ) if False else None,   # set to True once baseline is computed
        constraints=None,
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        monitor_schedule_name=f"{cfg.inference.endpoint_name}-dq-monitor",
    )
    logger.info("Data quality monitor scheduled.")


def setup_model_quality_monitor(cfg) -> None:
    session  = sagemaker.Session(boto_session=boto3.Session(region_name=cfg.aws.region))
    role_arn = cfg.aws.resolved_role_arn()

    monitor = ModelQualityMonitor(
        role=role_arn,
        instance_count=1,
        instance_type=cfg.aws.sagemaker.instance_type["processing"],
        sagemaker_session=session,
    )

    monitor.create_monitoring_schedule(
        endpoint_input=cfg.inference.endpoint_name,
        ground_truth_input=cfg.aws.s3_uri("artifacts", "ground-truth"),
        problem_type="BinaryClassification",
        output_s3_uri=cfg.aws.s3_uri("artifacts", "monitor-reports/model-quality"),
        schedule_cron_expression=CronExpressionGenerator.daily(),
        monitor_schedule_name=f"{cfg.inference.endpoint_name}-mq-monitor",
    )
    logger.info("Model quality monitor scheduled.")


def setup_cloudwatch_alarm(cfg) -> None:
    cw = boto3.client("cloudwatch", region_name=cfg.aws.region)
    cw.put_metric_alarm(
        AlarmName=f"{cfg.inference.endpoint_name}-high-latency",
        MetricName="ModelLatency",
        Namespace="AWS/SageMaker",
        Statistic="p99",
        Period=300,
        EvaluationPeriods=3,
        Threshold=2000,   # ms
        ComparisonOperator="GreaterThanThreshold",
        Dimensions=[
            {"Name": "EndpointName", "Value": cfg.inference.endpoint_name},
            {"Name": "VariantName",  "Value": "AllTraffic"},
        ],
        AlarmActions=[],   # TODO: add SNS topic ARN for PagerDuty / email
        TreatMissingData="notBreaching",
    )
    logger.info("CloudWatch latency alarm created.")


def main():
    cfg = get_config()
    enable_data_capture(cfg)
    setup_data_quality_monitor(cfg)
    setup_model_quality_monitor(cfg)
    setup_cloudwatch_alarm(cfg)
    logger.info("Monitoring setup complete.")


if __name__ == "__main__":
    main()
