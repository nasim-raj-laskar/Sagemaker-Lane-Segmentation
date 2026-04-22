"""
pipelines/pipeline.py
Defines the full SageMaker Pipeline:

  Step 1 — Processing   : Download from S3, preprocess, save train/val splits
  Step 2 — Training     : ResNet-UNet training on g4dn.4xlarge
  Step 3 — Evaluation   : Compute IoU, accuracy on hold-out set
  Step 4 — Condition    : Register if val_iou >= threshold
  Step 5 — Register     : Push to SageMaker Model Registry

Run with:
  python pipelines/pipeline.py --action upsert
  python pipelines/pipeline.py --action start
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import get_config
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


def get_pipeline(cfg=None) -> Pipeline:
    cfg = cfg or get_config()
    aws_cfg = cfg.aws

    session   = sagemaker.Session(boto_session=boto3.Session(region_name=aws_cfg.region))
    role_arn  = aws_cfg.resolved_role_arn()
    bucket    = aws_cfg.s3_bucket
    region    = aws_cfg.region

    # ── Pipeline parameters (can be overridden at execution time) ─────────────
    p_epochs        = ParameterInteger(name="Epochs",       default_value=cfg.training.epochs)
    p_batch_size    = ParameterInteger(name="BatchSize",    default_value=cfg.training.batch_size)
    p_lr            = ParameterFloat(  name="LearningRate", default_value=cfg.training.learning_rate)
    p_img_size      = ParameterInteger(name="ImgSize",      default_value=cfg.data.img_size)
    p_iou_threshold = ParameterFloat(  name="IoUThreshold", default_value=0.60)
    p_model_prefix  = ParameterString( name="ModelPrefix",  default_value=aws_cfg.s3_prefix.artifacts)

    # ── Step 1: Processing ────────────────────────────────────────────────────
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=aws_cfg.sagemaker.instance_type["processing"],
        instance_count=1,
        base_job_name="road-seg-preprocess",
        role=role_arn,
        sagemaker_session=session,
    )

    preprocess_step = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/{aws_cfg.s3_prefix.raw_images}",
                destination="/opt/ml/processing/input/image",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{aws_cfg.s3_prefix.raw_masks}",
                destination="/opt/ml/processing/input/mask",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/{aws_cfg.s3_prefix.processed}/train",
            ),
            ProcessingOutput(
                output_name="val",
                source="/opt/ml/processing/output/val",
                destination=f"s3://{bucket}/{aws_cfg.s3_prefix.processed}/val",
            ),
        ],
        code="scripts/preprocess_job.py",
        job_arguments=[
            "--img-size", p_img_size,
            "--val-split", str(cfg.data.val_split),
        ],
    )

    # ── Step 2: Training ──────────────────────────────────────────────────────
    tf_estimator = TensorFlow(
        entry_point="src/training/train.py",
        source_dir=".",
        role=role_arn,
        instance_count=1,
        instance_type=aws_cfg.sagemaker.instance_type["training"],
        framework_version="2.13",
        py_version="py310",
        hyperparameters={
            "epochs":        p_epochs,
            "batch-size":    p_batch_size,
            "learning-rate": p_lr,
            "img-size":      p_img_size,
        },
        metric_definitions=[
            {"Name": "val:iou",      "Regex": r"val_iou: (\S+)"},
            {"Name": "val:accuracy", "Regex": r"val_accuracy: (\S+)"},
            {"Name": "val:loss",     "Regex": r"val_loss: (\S+)"},
        ],
        output_path=f"s3://{bucket}/{aws_cfg.s3_prefix.artifacts}",
        checkpoint_s3_uri=f"s3://{bucket}/{aws_cfg.s3_prefix.artifacts}/checkpoints",
        sagemaker_session=session,
        environment={"MLFLOW_TRACKING_URI": aws_cfg.mlflow.get("tracking_uri", "")},
    )

    train_step = TrainingStep(
        name="TrainModel",
        estimator=tf_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="application/x-image",
            ),
            "val": sagemaker.inputs.TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri,
                content_type="application/x-image",
            ),
        },
    )

    # ── Step 3: Evaluation ────────────────────────────────────────────────────
    eval_processor = ScriptProcessor(
        command=["python3"],
        image_uri=f"{boto3.client('sts').get_caller_identity()['Account']}.dkr.ecr.{region}.amazonaws.com/{aws_cfg.ecr['training_image_repo']}:latest",
        instance_type=aws_cfg.sagemaker.instance_type["processing"],
        instance_count=1,
        base_job_name="road-seg-eval",
        role=role_arn,
        sagemaker_session=session,
    )

    eval_report = PropertyFile(
        name="EvalReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    eval_step = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/{aws_cfg.s3_prefix.processed}/val",
                destination="/opt/ml/processing/input/val",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/{aws_cfg.s3_prefix.artifacts}/evaluation",
            ),
        ],
        code="scripts/evaluate_job.py",
        property_files=[eval_report],
        job_arguments=["--img-size", p_img_size],
    )

    # ── Step 4: Conditional registration ─────────────────────────────────────
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=tf_estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["image/jpeg", "application/json"],
        response_types=["application/json"],
        inference_instances=[aws_cfg.sagemaker.instance_type["inference"]],
        transform_instances=[aws_cfg.sagemaker.instance_type["batch"]],
        model_package_group_name="RoadSegmentationModels",
        approval_status="PendingManualApproval" if cfg.pipeline.get("approval_required") else "Approved",
        model_metrics=sagemaker.model_metrics.ModelMetrics(
            model_statistics=sagemaker.model_metrics.MetricsSource(
                s3_uri=f"s3://{bucket}/{aws_cfg.s3_prefix.artifacts}/evaluation/evaluation.json",
                content_type="application/json",
            )
        ),
    )

    condition_step = ConditionStep(
        name="CheckIoUThreshold",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_report,
                    json_path="metrics.val_iou",
                ),
                right=p_iou_threshold,
            )
        ],
        if_steps=[register_step],
        else_steps=[],
    )

    # ── Assemble pipeline ─────────────────────────────────────────────────────
    pipeline = Pipeline(
        name=cfg.pipeline["name"],
        parameters=[
            p_epochs, p_batch_size, p_lr, p_img_size,
            p_iou_threshold, p_model_prefix,
        ],
        steps=[preprocess_step, train_step, eval_step, condition_step],
        sagemaker_session=session,
    )

    return pipeline


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Manage the SageMaker Pipeline")
    p.add_argument("--action", choices=["upsert", "start", "describe", "delete"], required=True)
    p.add_argument("--wait", action="store_true", help="Wait for pipeline execution to complete")
    args = p.parse_args()

    pipeline = get_pipeline()

    if args.action == "upsert":
        definition = pipeline.upsert(role_arn=get_config().aws.resolved_role_arn())
        logger.info("Pipeline upserted: %s", definition["PipelineArn"])

    elif args.action == "start":
        execution = pipeline.start()
        logger.info("Pipeline execution started: %s", execution.arn)
        if args.wait:
            execution.wait()
            logger.info("Execution complete. Status: %s", execution.describe()["PipelineExecutionStatus"])

    elif args.action == "describe":
        print(json.dumps(pipeline.describe(), indent=2, default=str))

    elif args.action == "delete":
        pipeline.delete()
        logger.info("Pipeline deleted.")


if __name__ == "__main__":
    main()
