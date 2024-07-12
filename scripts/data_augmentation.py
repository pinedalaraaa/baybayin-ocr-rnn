from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xml.etree.ElementTree as ET
import cv2
import os


# Data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest'
)

# Path to the directory containing the annotated data in XML format
cropped_images = "../Training Data/BA/cropped"

# Get a list of XML files and image files
image_files = sorted(os.listdir(cropped_images))


# Load images using ImageDataGenerator and apply data augmentation
for file in image_files:
    image = cv2.imread(os.path.join(cropped_images, file))

    # Convert image to RGB (required for data augmentation)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape image for flow method
    image = image.reshape((1,) + image.shape)
    i = 0
    for batch in datagen.flow(image, batch_size=1, save_to_dir='augmented', save_prefix='aug', save_format='png'):
        i += 1
        if i >= 5:  # Number of augmented images per original image
            break   # Exit loop after generating specified number of augmented images
