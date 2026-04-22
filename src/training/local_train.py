"""
src/training/local_train.py
Local training script that runs directly on the current GPU instance.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf

# Disable XLA for compatibility
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
tf.config.optimizer.set_jit(False)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config_loader import get_config, load_config
from src.data.preprocessing import build_dataset, split_dataset
from src.models.metrics import BinaryIoU, CombinedLoss
from src.models.resnet_unet import build_model_from_config
from src.utils.callbacks import build_callbacks
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Local GPU training")
    parser.add_argument("--data-dir", default="./dataset", help="Path to dataset directory")
    parser.add_argument("--output-dir", default="./models", help="Output directory for models")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    
    # Load config
    cfg = get_config() if not args.config else load_config(args.config)
    
    # Override config with CLI args
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.learning_rate:
        cfg.training.learning_rate = args.learning_rate
        
    logger.info(f"Training config: epochs={cfg.training.epochs}, batch_size={cfg.training.batch_size}, lr={cfg.training.learning_rate}")
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.warning("No GPU found, using CPU")
    
    # Setup data directories
    data_dir = Path(args.data_dir)
    img_dir = data_dir / "image"
    mask_dir = data_dir / "mask"
    
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Image or mask directory not found in {data_dir}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(cfg.aws.mlflow.get("experiment_name", "road-segmentation"))
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "img_size": cfg.data.img_size,
            "backbone": cfg.model.backbone,
            "loss": cfg.training.loss,
        })
        
        # Load and preprocess data
        logger.info("Loading dataset...")
        X, Y = build_dataset(
            image_dir=img_dir,
            mask_dir=mask_dir,
            img_size=cfg.data.img_size,
            augment=cfg.data.augmentation.get("enabled", True),
            aug_cfg=cfg.data.augmentation,
        )
        
        # Split dataset
        X_train, X_val, Y_train, Y_val = split_dataset(
            X, Y, cfg.data.val_split, cfg.data.random_seed
        )
        
        logger.info(f"Dataset split - Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Build model
        logger.info("Building model...")
        model = build_model_from_config(cfg.data, cfg.model)
        
        # Setup loss and optimizer
        loss_fn = CombinedLoss() if cfg.training.loss == "combined" else cfg.training.loss
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.training.learning_rate),
            loss=loss_fn,
            metrics=["accuracy", BinaryIoU(name="iou")],
        )
        
        # Setup callbacks
        callbacks = build_callbacks(
            cfg=cfg,
            model_dir=str(output_dir),
            output_dir=str(output_dir),
            mlflow_run_id=run.info.run_id,
        )
        
        # Train model
        logger.info("Starting training...")
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        
        # Log final metrics
        final_metrics = {k: float(v[-1]) for k, v in history.history.items()}
        mlflow.log_metrics(final_metrics)
        
        logger.info(f"Training complete. Final metrics: {json.dumps(final_metrics, indent=2)}")
        
        # Save model
        saved_model_path = output_dir / "road_seg_savedmodel"
        model.export(str(saved_model_path))
        logger.info(f"Model saved to {saved_model_path}")
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)
            
        # Log artifacts to MLflow
        mlflow.log_artifact(str(metrics_path))
        mlflow.tensorflow.log_model(model, artifact_path="model")
        
        logger.info(f"Training complete. Run ID: {run.info.run_id}")
        
        return final_metrics


if __name__ == "__main__":
    main()