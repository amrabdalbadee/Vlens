import streamlit as st
import cv2
import tempfile
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# Function to load YOLO model
@st.cache_resource
def load_model(onnx_model_path='runs/detect/custom_yolov8/weights/best.onnx'):
    return YOLO(onnx_model_path)

# Function to detect objects in an image
def detect_objects_image(model, img):
    # Perform object detection
    results = model(img)
    return results[0].plot()

# Function to detect objects in a video
def detect_objects_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # Display real-time frame
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)
        
    cap.release()

# Main App Code
def main():
    st.title("YOLO Real-Time Object Detection App")
    
    # Load YOLOv8 model (use your custom model if applicable)
    model = load_model('runs/detect/custom_yolov8/weights/best.onnx')  # Replace with your model path

    # Choose whether to upload image or video
    option = st.sidebar.selectbox("Choose Input Type", ("Image", "Video"))

    # For Image Input
    if option == "Image":
        uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_image is not None:
            # Save image to a temporary location
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_image.read())
            
            # Read and process image
            img = cv2.imread(tfile.name)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            
            # Perform object detection
            if st.button('Detect Objects'):
                st.write("Running YOLO detection on the image...")
                result_img = detect_objects_image(model, img)
                st.image(result_img, caption="Detection Result", use_column_width=True)

    # For Video Input
    elif option == "Video":
        uploaded_video = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'mkv'])
        if uploaded_video is not None:
            # Save video to a temporary location
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            
            # Perform real-time detection
            st.write("Running YOLO detection on the video...")
            detect_objects_video(model, tfile.name)

if __name__ == '__main__':
    main()
