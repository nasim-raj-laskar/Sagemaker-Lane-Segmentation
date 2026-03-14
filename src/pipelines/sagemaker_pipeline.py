"""src/pipelines/sagemaker_pipeline.py — full MLOps pipeline definition.

Stages:
    1. ProcessingStep  — data preprocessing (resize, split, save .npy)
    2. TrainingStep    — model training + MLflow logging
    3. EvaluationStep  — compute IoU / Dice on held-out test set
    4. RegisterStep    — register model in SageMaker Model Registry (conditional)
    5. DeployStep      — update real-time endpoint (conditional on approval)
"""

from __future__ import annotations

import boto3
from omegaconf import DictConfig
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_pipeline(cfg: DictConfig) -> Pipeline:
    """Build and return the SageMaker Pipeline object (not yet started)."""
    aws = cfg.aws
    role = aws.iam.execution_role_arn
    bucket = aws.s3.bucket
    region = aws.region

    # ── Pipeline parameters (overridable at execution time) ──────────────────
    p_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value=cfg.training.sagemaker.instance_type,
    )
    p_epochs = ParameterInteger(name="Epochs", default_value=cfg.training.epochs)
    p_min_iou = ParameterFloat(name="MinIoUThreshold", default_value=0.70)

    # ── Step 1: Processing ───────────────────────────────────────────────────
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="road-seg-preprocessing",
        role=role,
        sagemaker_session=_session(region),
    )
    processing_step = ProcessingStep(
        name="PreprocessKITTI",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/{aws.s3.data_prefix}/kitti",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="val", source="/opt/ml/processing/val"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code="scripts/processing_job.py",
    )

    # ── Step 2: Training ─────────────────────────────────────────────────────
    tf_estimator = TensorFlow(
        entry_point="src/training/train.py",
        source_dir=".",
        role=role,
        instance_type=p_instance_type,
        instance_count=cfg.training.sagemaker.instance_count,
        framework_version="2.15",
        py_version="py310",
        output_path=f"s3://{bucket}/{aws.s3.model_prefix}",
        base_job_name="road-seg-training",
        hyperparameters={"epochs": p_epochs},
        environment={"MLFLOW_TRACKING_URI": cfg.mlflow.tracking_uri},
        sagemaker_session=_session(region),
    )
    training_step = TrainingStep(
        name="TrainResNetUNet",
        estimator=tf_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            ),
            "val": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri
            ),
        },
    )

    # ── Step 3: Evaluation ───────────────────────────────────────────────────
    eval_processor = ScriptProcessor(
        command=["python3"],
        image_uri=f"{aws.ecr.uri}",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="road-seg-evaluation",
        role=role,
        sagemaker_session=_session(region),
    )
    eval_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        code="scripts/evaluate.py",
        property_files=[eval_report],
    )

    # ── Step 4: Register (conditional on IoU ≥ threshold) ───────────────────
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )
    register_step = RegisterModel(
        name="RegisterRoadSegModel",
        estimator=tf_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["image/jpeg", "image/png", "application/json"],
        response_types=["application/json"],
        inference_instances=["ml.g4dn.xlarge", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="RoadSegmentationModels",
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )
    condition_step = ConditionStep(
        name="CheckIoUThreshold",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(step_name=evaluation_step.name, property_file=eval_report, json_path="metrics.iou_score"),
                right=p_min_iou,
            )
        ],
        if_steps=[register_step],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=aws.pipeline.name,
        parameters=[p_instance_type, p_epochs, p_min_iou],
        steps=[processing_step, training_step, evaluation_step, condition_step],
        sagemaker_session=_session(region),
    )
    logger.info("Pipeline '%s' built with %d steps", aws.pipeline.name, 4)
    return pipeline


def _session(region: str):
    return __import__("sagemaker").Session(boto_session=boto3.Session(region_name=region))
