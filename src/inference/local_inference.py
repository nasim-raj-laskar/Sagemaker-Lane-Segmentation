"""
src/inference/local_inference.py
Local inference script for testing the trained model.
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config_loader import get_config, load_config
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class LocalPredictor:
    def __init__(self, model_path, config=None):
        self.cfg = config or get_config()
        
        # Import custom metrics before loading model
        from src.models.metrics import BinaryIoU
        
        # Try to load model in different formats
        model_path = Path(model_path)
        if model_path.suffix == '.keras':
            self.model = tf.keras.models.load_model(str(model_path), custom_objects={'BinaryIoU': BinaryIoU})
        elif model_path.suffix == '.h5':
            self.model = tf.keras.models.load_model(str(model_path), custom_objects={'BinaryIoU': BinaryIoU})
        elif model_path.is_dir():
            # SavedModel format - use TFSMLayer for Keras 3
            self.model = tf.keras.layers.TFSMLayer(str(model_path), call_endpoint='serving_default')
        else:
            # Try different extensions
            for ext in ['.keras', '.h5']:
                test_path = model_path.with_suffix(ext)
                if test_path.exists():
                    self.model = tf.keras.models.load_model(str(test_path), custom_objects={'BinaryIoU': BinaryIoU})
                    break
            else:
                # Fallback to TFSMLayer
                self.model = tf.keras.layers.TFSMLayer(str(model_path), call_endpoint='serving_default')
                
        self.img_size = self.cfg.data.img_size
        self.threshold = self.cfg.inference.threshold
        
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        return img
        
    def predict(self, image_path):
        """Run inference on a single image"""
        img = self.preprocess_image(image_path)
        
        # Run inference - handle both model types
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(img, verbose=0)
        else:
            # TFSMLayer case
            prediction = self.model(img)
            if isinstance(prediction, dict):
                # Extract the output tensor from dict
                prediction = list(prediction.values())[0]
        
        # Apply threshold
        if hasattr(prediction, 'numpy'):
            prediction = prediction.numpy()
        mask = (prediction[0] > self.threshold).astype(np.uint8) * 255
        
        return mask.squeeze()
        
    def predict_with_overlay(self, image_path, output_path=None):
        """Run inference and create overlay visualization"""
        # Load original image
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Get prediction
        mask = self.predict(image_path)
        
        # Resize mask to match original image
        mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
        
        # Create overlay
        overlay = original.copy()
        overlay_color = self.cfg.inference.overlay_color  # [B, G, R]
        
        # Apply colored mask where road is detected
        road_pixels = mask_resized > 0
        overlay[road_pixels] = overlay_color
        
        # Blend with original
        result = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
        
        if output_path:
            cv2.imwrite(str(output_path), result)
            logger.info(f"Overlay saved to {output_path}")
            
        return result, mask_resized


def main():
    parser = argparse.ArgumentParser(description="Local inference")
    parser.add_argument("--model-path", required=True, help="Path to saved model")
    parser.add_argument("--image-path", required=True, help="Path to input image")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    parser.add_argument("--config", help="Path to config file")
    args = parser.parse_args()
    
    setup_logging()
    
    # Setup paths
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load config
    cfg = get_config() if not args.config else load_config(args.config)
    
    # Create predictor
    predictor = LocalPredictor(str(model_path), cfg)
    
    # Run inference
    logger.info(f"Running inference on {image_path}")
    
    # Generate outputs
    mask_output = output_dir / f"{image_path.stem}_mask.png"
    overlay_output = output_dir / f"{image_path.stem}_overlay.png"
    
    # Get mask
    mask = predictor.predict(image_path)
    cv2.imwrite(str(mask_output), mask)
    
    # Get overlay
    overlay, _ = predictor.predict_with_overlay(image_path, overlay_output)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Mask: {mask_output}")
    logger.info(f"Overlay: {overlay_output}")


if __name__ == "__main__":
    main()