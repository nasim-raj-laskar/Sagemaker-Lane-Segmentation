# 🚗 Local GPU Pipeline Guide

This guide shows you how to run the Self-Driving Perceptron pipeline directly on your GPU instance instead of using SageMaker training jobs.

## 🏗️ Architecture (Local)

```
Local Dataset → Local Preprocessing → Local GPU Training → Local Inference
     ↓                                      ↓                    ↓
  dataset/                            models/              outputs/
                                         ↓
                                   MLflow Tracking
                                         ↓
                                  Optional S3 Upload
```

## 📋 Prerequisites

1. **GPU Instance**: You're already on a GPU instance (SageMaker Studio, EC2, etc.)
2. **Dataset**: Your KITTI dataset should be in `./dataset/` directory
3. **Python Environment**: Python 3.8+ with required packages

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install requirements
pip install -r requirements_local.txt

# Run setup script
bash setup_local.sh
```

### 2. Verify Dataset Structure

Ensure your dataset is organized like this:
```
dataset/
├── image/
│   ├── um_000000.png
│   ├── um_000001.png
│   └── ...
└── mask/
    ├── um_road_000000.png
    ├── um_road_000001.png
    └── ...
```

### 3. Run the Pipeline

**Full Pipeline (Recommended)**:
```bash
python run_local_pipeline.py
```

**Training Only**:
```bash
python run_local_pipeline.py --train-only
```

**Inference Only** (after training):
```bash
python run_local_pipeline.py --inference-only --model-path ./models/road_seg_savedmodel
```

## 🔧 Individual Components

### Training

```bash
# Direct training script
python src/training/local_train.py

# With custom parameters
python src/training/local_train.py --epochs 20 --batch-size 32 --learning-rate 0.001
```

### Inference

```bash
# Test on single image
python src/inference/local_inference.py \
    --model-path ./models/road_seg_savedmodel \
    --image-path ./dataset/image/um_000000.png \
    --output-dir ./outputs
```

## 📊 Monitoring & Results

### MLflow Tracking
- Local MLflow runs are stored in `./mlruns/`
- View results: `mlflow ui` (opens on http://localhost:5000)

### Output Files
- **Models**: `./models/road_seg_savedmodel/`
- **Metrics**: `./models/metrics.json`
- **Inference Results**: `./outputs/`
- **Logs**: Console output with structured logging

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
training:
  epochs: 10          # Adjust based on your needs
  batch_size: 16      # Adjust based on GPU memory
  learning_rate: 1.0e-4

data:
  img_size: 256       # Input image size
  val_split: 0.2      # Validation split ratio

model:
  backbone: resnet50  # ResNet backbone
  dropout_rate: 0.3   # Regularization
```

## 🐛 Troubleshooting

### Common Issues

**1. GPU Not Found**
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**2. Out of Memory**
- Reduce `batch_size` in config
- Reduce `img_size` if needed

**3. Dataset Not Found**
```bash
# Verify dataset structure
ls -la dataset/
ls -la dataset/image/ | head -5
ls -la dataset/mask/ | head -5
```

**4. Import Errors**
```bash
# Install missing packages
pip install -r requirements_local.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Performance Tips

1. **GPU Memory**: Monitor with `nvidia-smi`
2. **Batch Size**: Start with 8-16, increase if memory allows
3. **Image Size**: 256x256 is good balance of speed/quality
4. **Early Stopping**: Enabled by default to prevent overfitting

## 📈 Expected Results

After training, you should see:
- **IoU**: 0.6-0.8 (higher is better)
- **Accuracy**: 0.85-0.95
- **Training Time**: ~10-30 minutes (depends on epochs/data size)

## 🔄 Workflow

1. **Preprocessing**: Loads images, applies augmentation, splits data
2. **Training**: Trains ResNet-UNet model on local GPU
3. **Evaluation**: Computes metrics on validation set
4. **Inference**: Tests model on sample images
5. **Artifacts**: Saves model, metrics, and visualizations

## 📁 Output Structure

After running the pipeline:
```
Self-Driving_percerptron/
├── models/
│   ├── road_seg_savedmodel/     # Trained model
│   ├── metrics.json             # Training metrics
│   └── best_model.keras         # Best checkpoint
├── outputs/
│   ├── um_000000_result.png     # Inference results
│   ├── um_000001_result.png
│   └── ...
├── mlruns/                      # MLflow experiments
└── logs/                        # Training logs
```

## 🚀 Next Steps

1. **Experiment**: Try different hyperparameters
2. **Evaluate**: Check inference results in `./outputs/`
3. **Deploy**: Use the saved model for real-time inference
4. **Scale**: Upload artifacts to S3 for production use

## 💡 Tips

- Start with fewer epochs (5-10) for quick testing
- Monitor GPU usage with `nvidia-smi`
- Use MLflow UI to compare different runs
- Check sample outputs to verify model quality

---

**Need Help?** Check the logs for detailed error messages and ensure all prerequisites are met.