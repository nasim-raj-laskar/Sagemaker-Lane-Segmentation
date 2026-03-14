"""src/monitoring/monitor.py — CloudWatch metrics + Model Monitor setup."""

from __future__ import annotations

import json
from datetime import datetime

import boto3
from omegaconf import DictConfig
from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CloudWatchReporter:
    """Push custom metrics to CloudWatch from evaluation runs."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.namespace = cfg.monitoring.cloudwatch.namespace
        self.dimensions = [
            {"Name": k, "Value": v}
            for k, v in cfg.monitoring.cloudwatch.dimensions.items()
        ]
        self._cw = boto3.client("cloudwatch", region_name=cfg.aws.region)

    def put_metrics(self, metrics: dict[str, float], timestamp: datetime | None = None) -> None:
        """Publish a dict of metric_name → float to CloudWatch."""
        ts = timestamp or datetime.utcnow()
        metric_data = [
            {
                "MetricName": name,
                "Dimensions": self.dimensions,
                "Timestamp": ts,
                "Value": value,
                "Unit": "None",
            }
            for name, value in metrics.items()
        ]
        # CloudWatch accepts max 20 metrics per call
        for i in range(0, len(metric_data), 20):
            self._cw.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data[i : i + 20],
            )
        logger.info("Published %d metrics to CloudWatch namespace '%s'", len(metrics), self.namespace)

    def create_alarm(
        self,
        metric_name: str,
        threshold: float,
        comparison: str = "GreaterThanOrEqualToThreshold",
        evaluation_periods: int = 2,
        period_seconds: int = 300,
    ) -> None:
        alarm_name = f"road-seg-{metric_name.lower().replace('_', '-')}"
        self._cw.put_metric_alarm(
            AlarmName=alarm_name,
            Namespace=self.namespace,
            MetricName=metric_name,
            Dimensions=self.dimensions,
            Threshold=threshold,
            ComparisonOperator=comparison,
            EvaluationPeriods=evaluation_periods,
            Period=period_seconds,
            Statistic="Average",
            AlarmActions=[self.cfg.monitoring.model_monitor.alert_sns_arn],
            TreatMissingData="notBreaching",
        )
        logger.info("Alarm '%s' created (threshold=%.3f)", alarm_name, threshold)


class ModelMonitorSetup:
    """Configure SageMaker Model Monitor on a deployed endpoint."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        import sagemaker
        self._session = sagemaker.Session(
            boto_session=boto3.Session(region_name=cfg.aws.region)
        )

    def data_capture_config(self) -> DataCaptureConfig:
        dc = self.cfg.aws.endpoint.data_capture
        return DataCaptureConfig(
            enable_capture=dc.enabled,
            sampling_percentage=dc.sampling_percentage,
            destination_s3_uri=dc.s3_prefix,
        )

    def create_baseline(self, baseline_data_s3_uri: str) -> str:
        """Suggest constraints from a baseline dataset. Returns output S3 URI."""
        monitor = DefaultModelMonitor(
            role=self.cfg.aws.iam.execution_role_arn,
            instance_count=1,
            instance_type=self.cfg.aws.monitoring.instance_type,
            volume_size_in_gb=20,
            max_runtime_in_seconds=3600,
            sagemaker_session=self._session,
        )
        monitor.suggest_baseline(
            baseline_dataset=baseline_data_s3_uri,
            dataset_format=DatasetFormat.json(lines=True),
            output_s3_uri=self.cfg.aws.monitoring.baseline_s3_uri,
            wait=True,
            logs=False,
        )
        logger.info("Baseline created at %s", self.cfg.aws.monitoring.baseline_s3_uri)
        return self.cfg.aws.monitoring.baseline_s3_uri

    def attach_schedule(self, endpoint_name: str) -> None:
        """Attach a monitoring schedule to a running endpoint."""
        monitor = DefaultModelMonitor(
            role=self.cfg.aws.iam.execution_role_arn,
            instance_count=1,
            instance_type=self.cfg.aws.monitoring.instance_type,
            sagemaker_session=self._session,
        )
        monitor.create_monitoring_schedule(
            endpoint_input=endpoint_name,
            output_s3_uri=self.cfg.aws.monitoring.output_s3_uri,
            statistics=f"{self.cfg.aws.monitoring.baseline_s3_uri}/statistics.json",
            constraints=f"{self.cfg.aws.monitoring.baseline_s3_uri}/constraints.json",
            schedule_cron_expression=self.cfg.aws.monitoring.schedule,
            enable_cloudwatch_metrics=True,
        )
        logger.info("Monitoring schedule attached to endpoint '%s'", endpoint_name)
