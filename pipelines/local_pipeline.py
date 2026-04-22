"""
pipelines/local_pipeline.py
Local GPU pipeline that runs training directly on this instance.

Steps:
1. Preprocess data locally
2. Train model on local GPU
3. Evaluate model
4. Save artifacts to S3
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import get_config, load_config
from src.data.preprocessing import build_dataset, split_dataset
from src.models.metrics import BinaryIoU, CombinedLoss
from src.models.resnet_unet import build_model_from_config
from src.utils.callbacks import build_callbacks
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class LocalPipeline:
    def __init__(self, config_path=None):
        self.cfg = get_config() if not config_path else load_config(config_path)
        self.s3_client = boto3.client('s3', region_name=self.cfg.aws.region)
        self.bucket = self.cfg.aws.s3_bucket
        
    def run_preprocessing(self):
        """Step 1: Preprocess data locally"""
        logger.info("Starting data preprocessing...")
        
        # Use local dataset directory
        dataset_dir = Path("dataset")
        if not dataset_dir.exists():
            raise FileNotFoundError("Dataset directory not found. Expected: ./dataset/")
            
        img_dir = dataset_dir / "image"
        mask_dir = dataset_dir / "mask"
        
        if not img_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError("Image or mask directory not found in dataset/")
            
        # Build dataset
        X, Y = build_dataset(
            image_dir=img_dir,
            mask_dir=mask_dir,
            img_size=self.cfg.data.img_size,
            augment=self.cfg.data.augmentation.get("enabled", True),
            aug_cfg=self.cfg.data.augmentation,
        )
        
        # Split dataset
        X_train, X_val, Y_train, Y_val = split_dataset(
            X, Y, self.cfg.data.val_split, self.cfg.data.random_seed
        )
        
        logger.info(f"Dataset split - Train: {X_train.shape}, Val: {X_val.shape}")
        
        return X_train, X_val, Y_train, Y_val
        
    def run_training(self, X_train, X_val, Y_train, Y_val):
        """Step 2: Train model on local GPU"""
        logger.info("Starting model training...")
        
        # Setup MLflow
        if self.cfg.aws.mlflow.get("tracking_uri"):
            mlflow.set_tracking_uri(self.cfg.aws.mlflow["tracking_uri"])
        else:
            # Use local MLflow
            mlflow.set_tracking_uri("file:./mlruns")
            
        mlflow.set_experiment(self.cfg.aws.mlflow.get("experiment_name", "road-segmentation"))
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params({
                "epochs": self.cfg.training.epochs,
                "batch_size": self.cfg.training.batch_size,
                "learning_rate": self.cfg.training.learning_rate,
                "img_size": self.cfg.data.img_size,
                "backbone": self.cfg.model.backbone,
                "loss": self.cfg.training.loss,
            })
            
            # Build model
            model = build_model_from_config(self.cfg.data, self.cfg.model)
            
            # Setup loss and optimizer
            loss_fn = CombinedLoss() if self.cfg.training.loss == "combined" else self.cfg.training.loss
            
            model.compile(
                optimizer="adam",
                loss=loss_fn,
                metrics=["accuracy", BinaryIoU(name="iou")],
            )
            
            # Setup callbacks
            model_dir = Path("./models")
            model_dir.mkdir(exist_ok=True)
            
            callbacks = build_callbacks(
                cfg=self.cfg,
                model_dir=str(model_dir),
                output_dir="./outputs",
                mlflow_run_id=run.info.run_id,
            )
            
            # Train model
            history = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=self.cfg.training.epochs,
                batch_size=self.cfg.training.batch_size,
                callbacks=callbacks,
                verbose=1,
            )
            
            # Log metrics
            final_metrics = {k: float(v[-1]) for k, v in history.history.items()}
            mlflow.log_metrics(final_metrics)
            
            # Save model locally
            saved_model_path = model_dir / "road_seg_savedmodel.keras"
            model.save(str(saved_model_path))
            
            # Save metrics
            metrics_path = model_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(final_metrics, f, indent=2)
                
            mlflow.log_artifact(str(metrics_path))
            mlflow.tensorflow.log_model(model, artifact_path="model")
            
            logger.info(f"Training complete. Final IoU: {final_metrics.get('val_iou', 'N/A')}")
            
            return model, final_metrics, str(saved_model_path)
            
    def run_evaluation(self, model, X_val, Y_val):
        """Step 3: Evaluate model"""
        logger.info("Evaluating model...")
        
        predictions = model.predict(X_val)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score
        import numpy as np
        
        # Threshold predictions
        pred_binary = (predictions > self.cfg.inference.threshold).astype(int)
        y_val_binary = (Y_val > 0.5).astype(int)
        
        accuracy = accuracy_score(y_val_binary.flatten(), pred_binary.flatten())
        
        # Calculate IoU manually
        intersection = np.sum(pred_binary * y_val_binary)
        union = np.sum(pred_binary) + np.sum(y_val_binary) - intersection
        iou = intersection / (union + 1e-8)
        
        eval_metrics = {
            "accuracy": float(accuracy),
            "iou": float(iou),
        }
        
        logger.info(f"Evaluation metrics: {eval_metrics}")
        return eval_metrics
        
    def upload_artifacts(self, model_path, metrics):
        """Step 4: Upload artifacts to S3"""
        logger.info("Uploading artifacts to S3...")
        
        try:
            # Upload model
            model_s3_key = f"{self.cfg.aws.s3_prefix.artifacts}/road_seg_savedmodel.tar.gz"
            
            # Create tar.gz of model
            import tarfile
            tar_path = "road_seg_savedmodel.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(model_path, arcname="road_seg_savedmodel")
                
            self.s3_client.upload_file(tar_path, self.bucket, model_s3_key)
            
            # Upload metrics
            metrics_s3_key = f"{self.cfg.aws.s3_prefix.artifacts}/metrics.json"
            metrics_file = "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
                
            self.s3_client.upload_file(metrics_file, self.bucket, metrics_s3_key)
            
            logger.info(f"Artifacts uploaded to s3://{self.bucket}/{self.cfg.aws.s3_prefix.artifacts}/")
            
            # Cleanup
            os.remove(tar_path)
            os.remove(metrics_file)
            
        except Exception as e:
            logger.warning(f"Failed to upload to S3: {e}")
            
    def run_pipeline(self):
        """Run the complete local pipeline"""
        logger.info("Starting local GPU pipeline...")
        
        # Step 1: Preprocessing
        X_train, X_val, Y_train, Y_val = self.run_preprocessing()
        
        # Step 2: Training
        model, train_metrics, model_path = self.run_training(X_train, X_val, Y_train, Y_val)
        
        # Step 3: Evaluation
        eval_metrics = self.run_evaluation(model, X_val, Y_val)
        
        # Combine metrics
        all_metrics = {**train_metrics, **eval_metrics}
        
        # Step 4: Upload artifacts
        self.upload_artifacts(model_path, all_metrics)
        
        logger.info("Pipeline completed successfully!")
        return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Run local GPU pipeline")
    parser.add_argument("--config", help="Path to config file")
    args = parser.parse_args()
    
    setup_logging()
    
    pipeline = LocalPipeline(args.config)
    metrics = pipeline.run_pipeline()
    
    print(f"\nPipeline Results:")
    print(f"Final IoU: {metrics.get('val_iou', 'N/A'):.4f}")
    print(f"Final Accuracy: {metrics.get('val_accuracy', 'N/A'):.4f}")


if __name__ == "__main__":
    main()