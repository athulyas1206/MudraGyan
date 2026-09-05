import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import math

# Load the trained model
#model = load_model("asl_model.keras")
#model = load_model("mudra_cnn_model.keras")
model = load_model("mudra_cnn_model.keras")

# Define image size (300) and labels
imgSize = 300
labels = ['Musti', 'Pataka', 'Sikhara', 'Simhamukha', 'Trisula']

# Starting webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Prediction
        imgInput = imgWhite / 255.0  # Normalize
        imgInput = np.expand_dims(imgInput, axis=0)  # (1, 300, 300, 3)
        prediction = model.predict(imgInput)
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]

        # Show prediction
        text_position = (x, y - offset - 10) 
        cv2.putText(img, predicted_label, text_position , 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(img, (x - offset, y - offset), 
                      (x + w + offset, y + h + offset), (0, 0, 0), 3)

    cv2.imshow("ASL Test", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
