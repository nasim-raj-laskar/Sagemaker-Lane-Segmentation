#!/usr/bin/env python3

import boto3
import json
import os
from datetime import datetime

class ModelRegistry:
    def __init__(self, model_package_group_name="lane-segmentation-models"):
        region = os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION') or 'ap-south-1'
        
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.model_package_group_name = model_package_group_name
        self._ensure_model_package_group()
    
    def _ensure_model_package_group(self):
        """Create model package group if it doesn't exist"""
        try:
            self.sagemaker.describe_model_package_group(
                ModelPackageGroupName=self.model_package_group_name
            )
        except self.sagemaker.exceptions.ClientError:
            self.sagemaker.create_model_package_group(
                ModelPackageGroupName=self.model_package_group_name,
                ModelPackageGroupDescription="Lane segmentation model registry"
            )
    
    def register_model(self, model_s3_uri, metrics, accuracy_threshold=0.8):
        """Register model in SageMaker Model Registry"""
        # Determine approval status based on accuracy
        val_accuracy = metrics.get('final_val_accuracy', 0)
        approval_status = "Approved" if val_accuracy >= accuracy_threshold else "PendingManualApproval"
        
        # Get region for ECR image URI
        region = self.sagemaker.meta.region_name
        
        model_package_input = {
            'ModelPackageGroupName': self.model_package_group_name,
            'ModelPackageDescription': f"Lane segmentation model - Val Acc: {val_accuracy:.4f}",
            'ModelApprovalStatus': approval_status,
            'InferenceSpecification': {
                'Containers': [{
                    'Image': f'763104351884.dkr.ecr.{region}.amazonaws.com/tensorflow-inference:2.12-cpu',
                    'ModelDataUrl': model_s3_uri,
                    'Framework': 'TENSORFLOW'
                }],
                'SupportedContentTypes': ['application/json'],
                'SupportedResponseMIMETypes': ['application/json']
            },
            'ModelMetrics': {
                'ModelQuality': {
                    'Statistics': {
                        'ContentType': 'application/json',
                        'S3Uri': f"{model_s3_uri.rsplit('/', 1)[0]}/metrics.json"
                    }
                }
            }
        }
        
        response = self.sagemaker.create_model_package(**model_package_input)
        print(f"Model registered with status: {approval_status}")
        print(f"Model Package ARN: {response['ModelPackageArn']}")
        return response['ModelPackageArn']
    
    def get_latest_approved_model(self):
        """Get the latest approved model from registry"""
        try:
            response = self.sagemaker.list_model_packages(
                ModelPackageGroupName=self.model_package_group_name,
                ModelApprovalStatus='Approved',
                SortBy='CreationTime',
                SortOrder='Descending',
                MaxResults=1
            )
            
            if response['ModelPackageSummaryList']:
                model_package = response['ModelPackageSummaryList'][0]
                
                # Get model details
                details = self.sagemaker.describe_model_package(
                    ModelPackageName=model_package['ModelPackageArn']
                )
                
                model_s3_uri = details['InferenceSpecification']['Containers'][0]['ModelDataUrl']
                return model_s3_uri, model_package['ModelPackageArn']
            
        except Exception as e:
            print(f"Error getting approved model: {e}")
        
        return None, None
    
    def save_metrics_to_s3(self, metrics, s3_bucket, s3_key_prefix):
        """Save metrics as JSON to S3"""
        metrics_key = f"{s3_key_prefix}/metrics.json"
        
        try:
            self.s3.put_object(
                Bucket=s3_bucket,
                Key=metrics_key,
                Body=json.dumps(metrics, indent=2),
                ContentType='application/json'
            )
            print(f"Metrics saved to s3://{s3_bucket}/{metrics_key}")
        except Exception as e:
            print(f"Failed to save metrics: {e}")