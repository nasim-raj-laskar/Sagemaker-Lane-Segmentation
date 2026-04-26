#!/usr/bin/env python3

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker import get_execution_role
import yaml
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_config():
    with open('config/model.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    with open('config/train.yaml', 'r') as f:
        train_config = yaml.safe_load(f)
    return model_config, train_config

def launch_training_job():
    model_config, train_config = load_config()
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Get execution role
    try:
        role = get_execution_role()
    except ValueError:
        aws_account_id = os.getenv('AWS_ACCOUNT_ID')
        role_name = os.getenv('SAGEMAKER_ROLE', 'SageMakerExecutionRole')
        role = f"arn:aws:iam::{aws_account_id}:role/{role_name}"
        print(f"Using role: {role}")
    
    # Define the TensorFlow estimator
    tf_estimator = TensorFlow(
        entry_point='train.py',
        source_dir='src',
        role=role,
        instance_count=train_config['sagemaker']['instance_count'],
        instance_type=train_config['sagemaker']['instance_type'],
        framework_version=train_config['sagemaker']['framework_version'],
        py_version=train_config['sagemaker']['py_version'],
        script_mode=True,
        hyperparameters={
            'epochs': model_config['epochs'],
            'batch-size': model_config['batch_size'],
            'learning-rate': model_config['learning_rate'],
            'img-height': model_config['img_height'],
            'img-width': model_config['img_width'],
            'test-size': model_config['test_size'],
            'random-state': model_config['random_state'],
            'normalization-factor': model_config['normalization_factor'],
            'mask-threshold': model_config['mask_threshold'],
            'verbose': model_config['verbose'],
            'model-filename': model_config['model_filename'],
            'tar-filename': model_config['tar_filename'],
            's3-bucket': model_config['s3_bucket'],
            's3-model-prefix': model_config['s3_model_prefix'],
            'timestamp-format': model_config['timestamp_format']
        },
        output_path=f"s3://{train_config['s3']['bucket']}/{train_config['s3']['model_artifacts_path']}/",
        code_location=f"s3://{train_config['s3']['bucket']}/{train_config['s3']['code_location']}",
        base_job_name=train_config['training']['job_name_prefix']
    )
    
    training_input = f"s3://{train_config['s3']['bucket']}/{train_config['s3']['data_path']}"
    
    print("Starting training job...")
    print(f"Training data location: {training_input}")
    print(f"Instance type: {train_config['sagemaker']['instance_type']}")
    print(f"Framework: TensorFlow {train_config['sagemaker']['framework_version']}")
    
    # Start training
    tf_estimator.fit({'train': training_input})
    
    print("Training job completed!")
    print(f"Model artifacts location: {tf_estimator.model_data}")

if __name__ == '__main__':
    launch_training_job()