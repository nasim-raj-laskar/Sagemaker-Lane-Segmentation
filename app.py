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
    
    # FIXED VERSION

    mask = (prediction[0] > 0.5).astype(np.uint8).squeeze()
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    image_rgb = image.copy()

    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask == 1] = [0, 255, 0]

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
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as input_tmp:
            input_tmp.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(input_tmp.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            progress_bar = st.progress(0)
            frame_count = 0
            processed_frames = []
            
            # Process frames and collect them
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result_frame = process_image(frame, model)
                processed_frames.append(Image.fromarray(result_frame))
                
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
            
            cap.release()
            
            # Save as GIF for reliable web playback
            if processed_frames:
                gif_path = tempfile.mktemp(suffix='.gif')
                processed_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=processed_frames[1:],
                    duration=100,  # 100ms per frame
                    loop=0
                )
                
                st.subheader("Processed Video")
                st.image(gif_path)
                
                os.unlink(gif_path)
            
            os.unlink(input_tmp.name)

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