#!/usr/bin/env python3

import boto3
from src.model_registry import ModelRegistry

def list_models():
    """List all models in the registry with their status"""
    registry = ModelRegistry()
    
    try:
        response = registry.sagemaker.list_model_packages(
            ModelPackageGroupName=registry.model_package_group_name,
            SortBy='CreationTime',
            SortOrder='Descending'
        )
        
        print(f"\n=== Models in Registry: {registry.model_package_group_name} ===")
        
        if not response['ModelPackageSummaryList']:
            print("No models found in registry.")
            return
        
        for i, model in enumerate(response['ModelPackageSummaryList'], 1):
            print(f"\n{i}. Model Package: {model['ModelPackageArn']}")
            print(f"   Status: {model['ModelApprovalStatus']}")
            print(f"   Created: {model['CreationTime']}")
            
            # Get model details
            try:
                details = registry.sagemaker.describe_model_package(
                    ModelPackageName=model['ModelPackageArn']
                )
                model_uri = details['InferenceSpecification']['Containers'][0]['ModelDataUrl']
                print(f"   Model URI: {model_uri}")
                
                if 'ModelPackageDescription' in details:
                    print(f"   Description: {details['ModelPackageDescription']}")
                    
            except Exception as e:
                print(f"   Error getting details: {e}")
        
        # Show latest approved model
        print(f"\n=== Latest Approved Model ===")
        model_uri, model_arn = registry.get_latest_approved_model()
        if model_uri:
            print(f"Model URI: {model_uri}")
            print(f"Model ARN: {model_arn}")
        else:
            print("No approved models found.")
            
    except Exception as e:
        print(f"Error listing models: {e}")

def approve_model(model_package_arn):
    """Manually approve a model"""
    registry = ModelRegistry()
    
    try:
        registry.sagemaker.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus='Approved'
        )
        print(f"Model approved: {model_package_arn}")
    except Exception as e:
        print(f"Error approving model: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "approve":
        if len(sys.argv) > 2:
            approve_model(sys.argv[2])
        else:
            print("Usage: python model_registry_utils.py approve <model_package_arn>")
    else:
        list_models()