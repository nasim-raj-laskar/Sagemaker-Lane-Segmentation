import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import boto3
import subprocess
import os
import tempfile
import tarfile
import yaml
import threading
import time
from PIL import Image

st.set_page_config(page_title="Lane Segmentation", layout="wide")

@st.cache_resource
def load_model():
    """Load model from S3"""
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket='self-driving-perceptron', Prefix='model-artifacts/lane_segmentation_model')
        if 'Contents' in response:
            latest = max(response['Contents'], key=lambda x: x['LastModified'])
            s3_path = f"s3://self-driving-perceptron/{latest['Key']}"
            
            os.makedirs('models', exist_ok=True)
            s3.download_file('self-driving-perceptron', latest['Key'], 'models/latest.tar.gz')
            
            with tarfile.open('models/latest.tar.gz', 'r:gz') as tar:
                tar.extractall('models/latest')
            
            # Use TFSMLayer for SavedModel format in Keras 3
            model_layer = tf.keras.layers.TFSMLayer('models/latest/1', call_endpoint='serving_default')
            
            # Create a functional model wrapper
            inputs = tf.keras.Input(shape=(256, 832, 3))
            outputs = model_layer(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            return model, s3_path
    except Exception as e:
        st.error(f"Failed to load model from S3: {e}")
    return None, None

def preprocess_image(image):
    """Preprocess image for inference"""
    image = cv2.resize(image, (832, 256))
    return np.expand_dims(image.astype(np.float32) / 255.0, axis=0)

def process_image(image, model):
    """Process image and return result"""
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)
    
    # Handle TFSMLayer output (dictionary) or direct array
    if isinstance(prediction, dict):
        # Get the first (and likely only) output from the dictionary
        prediction = list(prediction.values())[0]
    
    # Create binary mask
    mask = (prediction[0] > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Ensure image is in RGB format
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if image is BGR and convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.max() > 1 else (image * 255).astype(np.uint8)
    else:
        image_rgb = image
    
    # Create colored overlay for lane detection (green for lanes)
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask == 1] = [0, 255, 0]  # Green color for detected lanes
    
    # Blend the original image with the colored mask
    result = cv2.addWeighted(image_rgb, 0.8, colored_mask, 0.2, 0)
    
    return result

def run_training_with_logs(epochs, log_container):
    """Update config and run training with real-time logs"""
    with open('config/model.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['epochs'] = epochs
    
    with open('config/model.yaml', 'w') as f:
        yaml.dump(config, f)
    
    process = subprocess.Popen(
        ['python', 'main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    logs = []
    for line in iter(process.stdout.readline, ''):
        if line:
            logs.append(line.strip())
            log_container.code('\n'.join(logs[-50:]))  # Show last 50 lines
            time.sleep(0.1)
    
    process.wait()
    return process.returncode == 0

st.title("Lane Segmentation Model")

# Load model
model, model_path = load_model()
if model is None:
    st.error("No model found. Please train a model first.")
else:
    st.success(f"Model loaded successfully from: {model_path}")

# Inference Section
st.header("Inference")
uploaded_file = st.file_uploader("Upload image or video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])

if uploaded_file and model:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == 'image':
        image = np.array(Image.open(uploaded_file))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image)
        
        with col2:
            st.subheader("Lane Segmentation")
            result = process_image(image, model)
            st.image(result)
    
    elif file_type == 'video':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tmp.name)
            frames = []
            
            progress_bar = st.progress(0)
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while frame_count < min(total_frames, 100):  # Limit frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                result_frame = process_image(frame, model)
                frames.append(result_frame)
                
                frame_count += 1
                progress_bar.progress(frame_count / min(total_frames, 100))
            
            cap.release()
            os.unlink(tmp.name)
            
            if frames:
                st.subheader("Processed Video Frames")
                for i, frame in enumerate(frames[::10]):  # Show every 10th frame
                    st.image(frame, caption=f"Frame {i*10}")

# Training Section
st.header("Train Model")

epochs = st.slider("Number of Epochs", min_value=1, max_value=50, value=15)

if st.button("Start Training"):
    st.info("Training started...")
    
    # Create log display container
    log_container = st.empty()
    
    # Run training in thread to allow real-time updates
    success = run_training_with_logs(epochs, log_container)
    
    if success:
        st.success("Training completed successfully!")
        st.cache_resource.clear()  # Clear model cache to reload
    else:
        st.error("Training failed!")