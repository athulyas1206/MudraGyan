import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
import numpy as np

# ------------------------------
# Parameters
# ------------------------------
folders = [
    "dataset/Pataka",
    "dataset/Musti",
    "dataset/Sikhara",
    "dataset/Simhamukha",
    "dataset/Trisula"
]

target_count = 500  # total images per class after augmentation
img_size = (300, 300)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

# ------------------------------
# Augmentation loop
# ------------------------------
for folder in folders:
    images = os.listdir(folder)
    current_count = len(images)
    print(f"Processing {folder}: {current_count} images")

    i = 0
    while current_count < target_count:
        img_name = images[i % len(images)]
        img_path = os.path.join(folder, img_name)

        # Load image
        img = load_img(img_path, target_size=img_size)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # reshape for datagen

        # Generate one augmented image at a time
        aug_iter = datagen.flow(x, batch_size=1)
        aug_img = next(aug_iter)[0].astype('uint8')

        # Save augmented image
        save_path = os.path.join(folder, f"aug_{current_count}_{img_name}")
        save_img(save_path, aug_img)

        current_count += 1
        i += 1

    print(f"Finished {folder}: {current_count} images")
