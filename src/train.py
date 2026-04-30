#!/usr/bin/env python3

import os
import tarfile
import tensorflow as tf
import boto3
import json
import mlflow
from datetime import datetime
from data_loader import DataLoader
from model import create_unet_model
from mlflow_config import setup_mlflow, log_metrics, log_params
from model_registry import ModelRegistry

def setup_gpu():
    """Configure GPU settings"""
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/conda'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def load_model_config():
    try:
        with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
            hyperparams = json.load(f)
    except:
        hyperparams = {}
    
    config = {
        'epochs': int(hyperparams.get('epochs', 15)),
        'batch_size': int(hyperparams.get('batch-size', 4)),
        'learning_rate': float(hyperparams.get('learning-rate', 1e-4)),
        'img_height': int(hyperparams.get('img-height', 256)),
        'img_width': int(hyperparams.get('img-width', 832)),
        'test_size': float(hyperparams.get('test-size', 0.2)),
        'random_state': int(hyperparams.get('random-state', 42)),
        'normalization_factor': float(hyperparams.get('normalization-factor', 255.0)),
        'mask_threshold': int(hyperparams.get('mask-threshold', 255)),
        'verbose': int(hyperparams.get('verbose', 1)),
        'model_filename': hyperparams.get('model-filename', 'model.keras').strip('"'),
        'tar_filename': hyperparams.get('tar-filename', 'model.tar.gz').strip('"'),
        's3_bucket': hyperparams.get('s3-bucket', 'self-driving-perceptron').strip('"'),
        's3_model_prefix': hyperparams.get('s3-model-prefix', 'model-artifacts/lane_segmentation_model').strip('"'),
        'timestamp_format': hyperparams.get('timestamp-format', '%Y%m%d_%H%M%S').strip('"'),
        'accuracy_threshold': float(hyperparams.get('accuracy-threshold', 0.8))
    }
    return config

def main():
    setup_gpu()
    config = load_model_config()
    
    # Setup MLflow tracking
    mlflow_enabled = setup_mlflow()
    
    if mlflow_enabled:
        # Start MLflow run
        with mlflow.start_run(run_name=f"lane_seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log hyperparameters
            log_params({
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'learning_rate': config['learning_rate'],
                'img_height': config['img_height'],
                'img_width': config['img_width']
            })
            
            run_training(config)
    else:
        run_training(config)

def run_training(config):
    
    # Data paths
    if os.environ.get('SM_CHANNEL_TRAIN'):
        image_dir = os.path.join(os.environ['SM_CHANNEL_TRAIN'], 'image')
        mask_dir = os.path.join(os.environ['SM_CHANNEL_TRAIN'], 'mask')
    else:
        image_dir = "/opt/ml/input/data/train/image"
        mask_dir = "/opt/ml/input/data/train/mask"
    
    print(f"Loading data from {image_dir} and {mask_dir}")
    
    # Load and prepare data
    data_loader = DataLoader(image_dir, mask_dir, config)
    X, Y = data_loader.load_data()
    X_train, X_val, Y_train, Y_val = data_loader.split_data(X, Y)
    
    print(f"Training data shape: {X_train.shape}, {Y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {Y_val.shape}")
    
    # Create model
    model = create_unet_model(config)
    
    # Train model
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=config['verbose']
    )
    
    # Log training metrics to MLflow
    for epoch in range(len(history.history['loss'])):
        log_metrics({
            'train_loss': history.history['loss'][epoch],
            'val_loss': history.history['val_loss'][epoch],
            'train_accuracy': history.history['binary_accuracy'][epoch],
            'val_accuracy': history.history['val_binary_accuracy'][epoch]
        }, step=epoch)
    
    # Save model in SageMaker serving format
    model_dir = os.environ.get('SM_MODEL_DIR', "/opt/ml/model")
    serving_model_dir = os.path.join(model_dir, "1")
    os.makedirs(serving_model_dir, exist_ok=True)
    
    # Save as SavedModel format for serving
    model.save(serving_model_dir, save_format='tf')
    
    # Create tar.gz for SageMaker (only include the serving format)
    tar_path = os.path.join(model_dir, config['tar_filename'])
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(serving_model_dir, arcname="1", recursive=True)
    
    print(f"Model saved to {serving_model_dir} (serving format)")
    print(f"Model archive saved to {tar_path}")
    
    # Get next version number
    s3_client = boto3.client('s3')
    version = get_next_version(s3_client, config['s3_bucket'], config['s3_model_prefix'])
    
    # Upload model to S3 with versioning
    timestamp = datetime.now().strftime(config['timestamp_format'])
    s3_key = f"{config['s3_model_prefix']}/v{version}/{timestamp}.tar.gz"
    
    try:
        s3_client.upload_file(tar_path, config['s3_bucket'], s3_key)
        print(f"Model uploaded to s3://{config['s3_bucket']}/{s3_key}")
        model_s3_uri = f"s3://{config['s3_bucket']}/{s3_key}"
    except Exception as e:
        print(f"Failed to upload model to S3: {e}")
        return
    
    # Prepare final metrics
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_acc = history.history['binary_accuracy'][-1]
    final_val_acc = history.history['val_binary_accuracy'][-1]
    
    metrics = {
        'final_train_loss': float(final_loss),
        'final_val_loss': float(final_val_loss),
        'final_train_accuracy': float(final_acc),
        'final_val_accuracy': float(final_val_acc),
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'learning_rate': config['learning_rate'],
        'timestamp': timestamp,
        'version': version
    }
    
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final training accuracy: {final_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    
    # Log final metrics to MLflow
    log_metrics(metrics)
    
    # Save metrics to S3 and register model
    try:
        registry = ModelRegistry()
        
        # Save metrics to S3
        s3_key_prefix = f"{config['s3_model_prefix']}/v{version}"
        registry.save_metrics_to_s3(metrics, config['s3_bucket'], s3_key_prefix)
        
        # Register model in SageMaker Model Registry
        model_package_arn = registry.register_model(
            model_s3_uri, 
            metrics, 
            config['accuracy_threshold']
        )
        
        print(f"Model registered: {model_package_arn}")
        
    except Exception as e:
        print(f"Failed to register model: {e}")

def get_next_version(s3_client, bucket, prefix):
    """Get next version number for model versioning"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{prefix}/v",
            Delimiter='/'
        )
        
        versions = []
        if 'CommonPrefixes' in response:
            for obj in response['CommonPrefixes']:
                version_str = obj['Prefix'].split('/')[-2]  # Extract version from path
                if version_str.startswith('v'):
                    try:
                        versions.append(int(version_str[1:]))
                    except ValueError:
                        continue
        
        return max(versions) + 1 if versions else 1
    except Exception:
        return 1

if __name__ == '__main__':
    main()