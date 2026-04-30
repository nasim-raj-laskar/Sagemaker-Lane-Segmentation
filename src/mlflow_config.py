#!/usr/bin/env python3

import os
import mlflow
from dotenv import load_dotenv

def setup_mlflow(mlflow_arn=None):
    """Setup MLflow tracking with SageMaker MLflow server or local"""
    if not mlflow_arn:
        load_dotenv()
        mlflow_arn = os.getenv('MLFLOW_ARN')
    
    if mlflow_arn:
        # Extract region and app name from ARN
        arn_parts = mlflow_arn.split(':')
        region = arn_parts[3]
        app_name = arn_parts[-1].split('/')[-1]
        
        # Set MLflow tracking URI for SageMaker MLflow
        tracking_uri = f"https://{app_name}.{region}.aws"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI set to: {tracking_uri}")
        return True
    else:
        # Use local MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        print("Using local MLflow tracking")
        return True

def log_metrics(metrics_dict, step=None):
    """Log metrics to MLflow"""
    for key, value in metrics_dict.items():
        mlflow.log_metric(key, value, step=step)

def log_params(params_dict):
    """Log parameters to MLflow"""
    mlflow.log_params(params_dict)