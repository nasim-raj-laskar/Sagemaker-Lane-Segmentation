"""
scripts/launch_training_job.py
Launches a standalone SageMaker Training Job directly.
No pipeline, no Docker build needed — uses the official TF container.
Run: python scripts/launch_training_job.py
"""
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from config.config_loader import get_config

cfg     = get_config()
session = sagemaker.Session(
    boto_session=boto3.Session(region_name=cfg.aws.region)
)
role_arn = cfg.aws.resolved_role_arn()

estimator = TensorFlow(
    entry_point="train.py",
    source_dir="src/training",          # SM zips this folder and uploads it
    role=role_arn,
    instance_count=1,
    instance_type=cfg.aws.sagemaker.instance_type["training"],  # ml.g4dn.4xlarge
    framework_version="2.13",
    py_version="py310",
    hyperparameters={
        "epochs":       10,             # start low for first SM job test
        "batch-size":   8,
        "learning-rate": 1e-4,
        "img-size":     256,
    },
    output_path=f"s3://{cfg.aws.s3_bucket}/{cfg.aws.s3_prefix.artifacts}",
    base_job_name="road-seg-training",
    sagemaker_session=session,
)

# Point at your S3 data
estimator.fit(
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{cfg.aws.s3_bucket}/{cfg.aws.s3_prefix.raw_images}",
            content_type="application/x-image",
        )
    },
    wait=True,    # streams logs to your terminal in real time
    logs=True,
)

print("Training job complete.")
print("Model artifact at:", estimator.model_data)