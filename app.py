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
    """Load model from local or S3"""
    # Try local first
    for path in ['models/model.keras', 'models/model_local.keras']:
        if os.path.exists(path):
            return tf.keras.models.load_model(path, compile=False)
    
    # Try S3
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket='self-driving-perceptron', Prefix='model-artifacts/lane_segmentation_model')
        if 'Contents' in response:
            latest = max(response['Contents'], key=lambda x: x['LastModified'])
            
            os.makedirs('models', exist_ok=True)
            s3.download_file('self-driving-perceptron', latest['Key'], 'models/latest.tar.gz')
            
            with tarfile.open('models/latest.tar.gz', 'r:gz') as tar:
                tar.extractall('models/latest')
            
            return tf.keras.models.load_model('models/latest/1', compile=False)
    except:
        pass
    return None

def preprocess_image(image):
    """Preprocess image for inference"""
    image = cv2.resize(image, (832, 256))
    return np.expand_dims(image.astype(np.float32) / 255.0, axis=0)

def process_image(image, model):
    """Process image and return result"""
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)
    
    mask = (prediction[0] > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    overlay = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

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
model = load_model()
if model is None:
    st.error("No model found. Please train a model first.")
else:
    st.success("Model loaded successfully!")

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