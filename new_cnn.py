import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Load and preprocess dataset
# -----------------------------
data_dir = "dataset"  # Folder with subfolders A, B, C...
labels = []
images = []

class_names = sorted(os.listdir(data_dir))  # e.g., ['A', 'B', 'C']

# Check image count per class
for class_name in class_names:
    folder_path = os.path.join(data_dir, class_name)
    print(f"{class_name}: {len(os.listdir(folder_path))} images")

# Load images
for idx, class_name in enumerate(class_names):
    folder_path = os.path.join(data_dir, class_name)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (300, 300))  # ✅ Ensure consistent size
            images.append(img)
            labels.append(idx)

# Convert to arrays
X = np.array(images).astype('float32') / 255.0  # Normalize
y = to_categorical(labels, num_classes=len(class_names))  # One-hot encode

# Shuffle dataset
X, y = shuffle(X, y, random_state=42)

# -----------------------------
# 2. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Data Augmentation
# -----------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# -----------------------------
# 4. Build CNN model
# -----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

# -----------------------------
# 5. Compile model
# -----------------------------
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# 6. Early stopping
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# -----------------------------
# 7. Train model
# -----------------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -----------------------------
# 8. Save model
# -----------------------------
model.save("mudra_cnn_model.keras")  # Saved in modern format

print("✅ Training complete and model saved as mudra_model.keras")

import pickle
with open("saved_model/history.pkl", "wb") as f:
    pickle.dump(history.history, f)