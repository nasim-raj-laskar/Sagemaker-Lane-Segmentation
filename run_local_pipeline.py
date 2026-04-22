#!/usr/bin/env python3
"""
run_local_pipeline.py
Main script to run the complete local GPU pipeline.
"""
import os
# Set correct CUDA paths and suppress warnings
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/conda'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import logging
import sys
from pathlib import Path

import tensorflow as tf
# Disable XLA JIT compilation for stability
tf.config.optimizer.set_jit(False)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipelines.local_pipeline import LocalPipeline
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def check_requirements():
    """Check if all requirements are met"""
    # Check dataset
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        logger.error("Dataset directory not found. Please ensure ./dataset/ exists with image/ and mask/ subdirectories")
        return False
        
    img_dir = dataset_dir / "image"
    mask_dir = dataset_dir / "mask"
    
    if not img_dir.exists() or not mask_dir.exists():
        logger.error("Image or mask directory not found in dataset/")
        return False
        
    # Check if there are files
    img_files = list(img_dir.glob("*.png"))
    mask_files = list(mask_dir.glob("*.png"))
    
    if not img_files or not mask_files:
        logger.error("No PNG files found in dataset directories")
        return False
        
    logger.info(f"Found {len(img_files)} images and {len(mask_files)} masks")
    
    # Check GPU
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logger.info(f"Found {len(gpus)} GPU(s)")
        else:
            logger.warning("No GPU found, will use CPU (slower)")
    except Exception as e:
        logger.warning(f"Could not check GPU: {e}")
        
    return True


def main():
    parser = argparse.ArgumentParser(description="Run local GPU pipeline")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--train-only", action="store_true", help="Only run training")
    parser.add_argument("--inference-only", action="store_true", help="Only run inference")
    parser.add_argument("--model-path", help="Path to model for inference")
    parser.add_argument("--test-image", help="Path to test image for inference")
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info("🚗 Starting Self-Driving Perceptron Local Pipeline")
    
    # Check requirements
    if not check_requirements():
        logger.error("Requirements check failed. Please fix the issues above.")
        sys.exit(1)
    
    if args.inference_only:
        # Run inference only
        if not args.model_path:
            logger.error("--model-path required for inference-only mode")
            sys.exit(1)
            
        from src.inference.local_inference import LocalPredictor
        from config.config_loader import get_config
        
        cfg = get_config()
        predictor = LocalPredictor(args.model_path, cfg)
        
        if args.test_image:
            test_image = Path(args.test_image)
            if test_image.exists():
                logger.info(f"Running inference on {test_image}")
                output_dir = Path("./outputs")
                output_dir.mkdir(exist_ok=True)
                
                overlay, mask = predictor.predict_with_overlay(
                    test_image, 
                    output_dir / f"{test_image.stem}_result.png"
                )
                logger.info("Inference complete!")
            else:
                logger.error(f"Test image not found: {test_image}")
        else:
            # Test on first image from dataset
            dataset_dir = Path("dataset/image")
            test_images = list(dataset_dir.glob("*.png"))[:3]  # Test on first 3 images
            
            output_dir = Path("./outputs")
            output_dir.mkdir(exist_ok=True)
            
            for img_path in test_images:
                logger.info(f"Testing on {img_path.name}")
                overlay, mask = predictor.predict_with_overlay(
                    img_path,
                    output_dir / f"{img_path.stem}_result.png"
                )
            
            logger.info(f"Inference complete! Results saved to {output_dir}")
            
    elif args.train_only:
        # Run training only
        from src.training.local_train import main as train_main
        train_main()
        
    else:
        # Run full pipeline
        pipeline = LocalPipeline(args.config)
        metrics = pipeline.run_pipeline()
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   IoU: {metrics.get('val_iou', 'N/A'):.4f}")
        logger.info(f"   Accuracy: {metrics.get('val_accuracy', 'N/A'):.4f}")
        
        # Test inference on a sample image
        model_path = Path("./models/best_checkpoint.keras")
        if model_path.exists():
            logger.info("🔍 Testing inference on sample images...")
            
            from src.inference.local_inference import LocalPredictor
            from config.config_loader import get_config
            
            cfg = get_config()
            predictor = LocalPredictor(str(model_path), cfg)
            
            # Test on first few images
            dataset_dir = Path("dataset/image")
            test_images = list(dataset_dir.glob("*.png"))[:3]
            
            output_dir = Path("./outputs")
            output_dir.mkdir(exist_ok=True)
            
            for img_path in test_images:
                logger.info(f"Testing on {img_path.name}")
                overlay, mask = predictor.predict_with_overlay(
                    img_path,
                    output_dir / f"{img_path.stem}_result.png"
                )
            
            logger.info(f"✅ Sample results saved to {output_dir}")


if __name__ == "__main__":
    main()