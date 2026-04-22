#!/bin/bash
# setup_local.sh
# Setup script for local GPU pipeline

echo "🚗 Setting up Self-Driving Perceptron Local Pipeline..."

# Make scripts executable
chmod +x run_local_pipeline.py
chmod +x src/training/local_train.py
chmod +x src/inference/local_inference.py

# Create necessary directories
mkdir -p models
mkdir -p outputs
mkdir -p mlruns

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Ensure your dataset is in ./dataset/ with image/ and mask/ subdirectories"
echo "2. Install requirements: pip install -r requirements_local.txt"
echo "3. Run the pipeline: python run_local_pipeline.py"
echo ""
echo "🔧 Available commands:"
echo "  python run_local_pipeline.py                    # Run full pipeline"
echo "  python run_local_pipeline.py --train-only       # Train only"
echo "  python run_local_pipeline.py --inference-only --model-path ./models/road_seg_savedmodel"
echo "  python src/training/local_train.py              # Direct training"
echo "  python src/inference/local_inference.py --model-path ./models/road_seg_savedmodel --image-path ./dataset/image/um_000000.png"