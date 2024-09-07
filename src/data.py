def filter_classes(original_dir, target_dir, classes):
    """
    Filters and copies only the specified classes from the original dataset directory to a new directory.

    Parameters:
        original_dir (str): Path to the original dataset directory containing all classes.
        target_dir (str): Path to the target directory where filtered classes will be copied.
        classes (list): List of class names to include (e.g., ['cars', 'buses', 'trucks']).
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    for class_name in classes:
        class_dir = os.path.join(original_dir, class_name)
        if os.path.exists(class_dir):
            shutil.copytree(class_dir, os.path.join(target_dir, class_name))
        else:
            print(f"Class directory {class_name} does not exist in {original_dir}. Skipping.")

def load_filtered_dataset(classes, train_dir, val_dir, test_dir, target_dir, img_size, batch_size):
    """
    Loads the dataset by filtering out only the specified classes from the dataset directories.

    Parameters:
        train_dir (str): Path to the original training dataset directory.
        val_dir (str): Path to the original validation dataset directory.
        test_dir (str): Path to the original test dataset directory.
        target_dir (str): Path to the directory where filtered classes will be copied.
        img_size (tuple): Target size for images (height, width).
        batch_size (int): Batch size for the data generator.

    Returns:
        train_generator, val_generator, test_generator: Data generators for the filtered dataset.
    """
    # Create filtered directories
    filtered_train_dir = os.path.join(target_dir, 'train')
    filtered_val_dir = os.path.join(target_dir, 'val')
    filtered_test_dir = os.path.join(target_dir, 'test')
    
    filter_classes(train_dir, filtered_train_dir, classes)
    filter_classes(val_dir, filtered_val_dir, classes)
    filter_classes(test_dir, filtered_test_dir, classes)
    
    # Load filtered dataset
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        filtered_train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_test_datagen.flow_from_directory(
        filtered_val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = val_test_datagen.flow_from_directory(
        filtered_test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, val_generator, test_generator

