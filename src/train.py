#!/usr/bin/env python3

import os
import tarfile
import tensorflow as tf
import boto3
import json
from datetime import datetime
from data_loader import DataLoader
from model import create_unet_model

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
        'model_filename': hyperparams.get('model-filename', 'model.keras'),
        'tar_filename': hyperparams.get('tar-filename', 'model.tar.gz'),
        's3_bucket': hyperparams.get('s3-bucket', 'self-driving-perceptron'),
        's3_model_prefix': hyperparams.get('s3-model-prefix', 'model-artifacts/lane_segmentation_model'),
        'timestamp_format': hyperparams.get('timestamp-format', '%Y%m%d_%H%M%S')
    }
    return config

def main():
    setup_gpu()
    config = load_model_config()
    
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
    
    # Save model
    model_dir = os.environ.get('SM_MODEL_DIR', "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, config['model_filename'])
    model.save(model_path)
    
    # Create tar.gz for SageMaker
    tar_path = os.path.join(model_dir, config['tar_filename'])
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname=config['model_filename'])
    
    print(f"Model saved to {model_path}")
    print(f"Model archive saved to {tar_path}")
    
    # Upload model to S3 with timestamp
    timestamp = datetime.now().strftime(config['timestamp_format'])
    s3_client = boto3.client('s3')
    s3_key = f"{config['s3_model_prefix']}_{timestamp}.tar.gz"
    
    try:
        s3_client.upload_file(tar_path, config['s3_bucket'], s3_key)
        print(f"Model uploaded to s3://{config['s3_bucket']}/{s3_key}")
    except Exception as e:
        print(f"Failed to upload model to S3: {e}")
    
    # Print final metrics
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_acc = history.history['binary_accuracy'][-1]
    final_val_acc = history.history['val_binary_accuracy'][-1]
    
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final training accuracy: {final_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")

if __name__ == '__main__':
    main()