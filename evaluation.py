# evaluation.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import cv2

# -------------------------------
# 1. Parameters
# -------------------------------
MODEL_PATH = "mudra_cnn_model.keras"  # trained model
TEST_FOLDERS = sorted([
       "test_data/Musti", 
       "test_data/Pataka",
       "test_data/Sikhara", 
       "test_data/Simhamukha", 
       "test_data/Trisula" 
])
IMG_SIZE = 300

# Folder to save misclassified images
REVIEW_DIR = "misclassified_review"
os.makedirs(REVIEW_DIR, exist_ok=True)

# -------------------------------
# 2. Prepare class labels
# -------------------------------
class_labels = [os.path.basename(f) for f in TEST_FOLDERS]

# -------------------------------
# 3. Load Model
# -------------------------------
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------------
# 4. Load test images
# -------------------------------
X_test = []
y_test = []
img_names = []

for idx, folder in enumerate(TEST_FOLDERS):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    for f in files:
        img = load_img(f, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
        X_test.append(img_array)
        y_test.append(idx)
        img_names.append(f)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Convert to categorical
from tensorflow.keras.utils import to_categorical
y_test_cat = to_categorical(y_test, num_classes=len(TEST_FOLDERS))

# -------------------------------
# 5. Evaluate model
# -------------------------------
results = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"\nTest Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]*100:.2f}%")

# -------------------------------
# 6. Predictions
# -------------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# -------------------------------
# 7. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes, target_names=class_labels))

# Per-class Accuracy
class_correct = np.diag(cm)
class_total = cm.sum(axis=1)
class_acc = class_correct / class_total

plt.figure(figsize=(10,6))
sns.barplot(x=class_labels, y=class_acc)
plt.title("Per-class Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# -------------------------------
# 8. Show misclassified images
# -------------------------------
misclassified_idx = np.where(y_pred_classes != y_test)[0]
max_per_class = 5
shown_per_class = {i: 0 for i in range(len(class_labels))}

plt.figure(figsize=(15, 10))
count = 1

# CSV file to log misclassified images
csv_file = os.path.join(REVIEW_DIR, "misclassified_images.csv")
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Image_Name", "True_Label", "Predicted_Label"])

    for idx in misclassified_idx:
        true_class = y_test[idx]
        pred_class = y_pred_classes[idx]

        # Save image in review folder
        img_array = (X_test[idx] * 255).astype(np.uint8)
        save_name = f"{class_labels[true_class]}_pred_{class_labels[pred_class]}_{idx}.jpg"
        save_path = os.path.join(REVIEW_DIR, save_name)
        cv2.imwrite(save_path, img_array)

        # Log in CSV
        writer.writerow([os.path.basename(img_names[idx]), class_labels[true_class], class_labels[pred_class]])

        # Plot a few images
        if shown_per_class[true_class] < max_per_class:
            plt.subplot(len(class_labels), max_per_class, count)
            plt.imshow(array_to_img(X_test[idx]))
            plt.title(f"True: {class_labels[true_class]}\nPred: {class_labels[pred_class]}")
            plt.axis('off')
            shown_per_class[true_class] += 1
            count += 1

plt.suptitle("Some Misclassified Mudra Images")
plt.show()

print(f"\n✅ Misclassified images saved to '{REVIEW_DIR}' and CSV report generated.")
