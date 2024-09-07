import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import shutil
from ultralytics import YOLO
import cv2

# Function to create the classification model
def create_classification_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False)
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Classification head
    ])
    
    return model

# Function to fine-tune and train the model
def train_model(model, train_generator, val_generator, epochs, batch_size):
    
    #checkpoint = ModelCheckpoint('ssd_vehicle_classification.weights.h5', monitor='val_loss', save_best_only=True)
    checkpoint = ModelCheckpoint('ssd_vehicle_classification.model.keras', monitor='val_loss', save_best_only=True)
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint]
    )
    
    return history

# Function to evaluate the model on the test dataset
def evaluate_model(model, test_generator):
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

def load_model(model_path='yolov8n.pt'):
    """
    Loads the YOLOv8 model.
    
    Args:
        model_path (str): Path to the trained YOLOv8 model.

    Returns:
        model: Loaded YOLO model.
    """
    model = YOLO(model_path)
    return model

def predict_image(model, image_path, save_path='result_image.jpg'):
    """
    Performs object detection on a single image.
    
    Args:
        model: Loaded YOLOv8 model.
        image_path (str): Path to the input image.
        save_path (str): Path to save the result image with bounding boxes.

    Returns:
        results: YOLO detection results.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Perform inference
    results = model(image)
    
    # Save and display the result
    results[0].save(save_path)
    return results

def predict_video(model, video_path, output_path='output_video.mp4'):
    """
    Performs object detection on a video.

    Args:
        model: Loaded YOLOv8 model.
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video with bounding boxes.

    Returns:
        None
    """
    # Capture video from file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define video writer to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference on each frame
        results = model(frame)

        # Get the annotated frame
        annotated_frame = results[0].plot()

        # Write the frame into the output video
        out.write(annotated_frame)
        
    # Release resources
    cap.release()
    out.release()
