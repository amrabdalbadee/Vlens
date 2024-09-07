

# Main script execution
if __name__ == "__main__":
    
    # IMG_SIZE = (224, 224)
    # BATCH_SIZE = 32
    # EPOCHS = 10
    # NUM_CLASSES = 3  # Cars, Buses, Trucks
    # # Load and preprocess the dataset
    # train_gen, val_gen, test_gen = load_filtered_dataset(classes, TRAIN_DIR, VAL_DIR, TEST_DIR, TARGET_DIR, IMG_SIZE, BATCH_SIZE)    
    # # Create the SSD model
    # ssd_model = create_ssd_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES)
    
    # # Fine-tune and train the model
    # train_model(ssd_model, train_gen, val_gen, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # # Evaluate the model on the test dataset
    # evaluate_model(ssd_model, test_gen)

    model = load_model('../runs/detect/custom_yolov8/weights/best.pt')  # or use your custom model
    results = predict_image(model, '../dataset/test/bus/1.png', save_path='result_image.jpg')
    predict_video(model, '../test/input/test.mp4', output_path='../test/results/result_video.mp4')
    