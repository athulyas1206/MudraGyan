import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0) # for accesing the web cam 0 is the index for webcam
detector = HandDetector(maxHands=1)

offset=20
imgSize=300

folder ='test_data/Trisula'
counter=0

while True:
    success, img = cap.read()
    hands, img =detector.findHands(img)

    if hands:
        hand=hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape
 
        

        aspectRatio = h/w

        if aspectRatio <1:
            k= imgSize/h
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hGap + min(hCal, imgWhite.shape[0] - hGap), :] = imgResize[:min(hCal, imgWhite.shape[0] - hGap), :]

        else:
            k= imgSize/w
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wGap + min(wCal, imgWhite.shape[1] - wGap)] = imgResize[:, :min(wCal, imgWhite.shape[1] - wGap)]





        cv2.imshow('ImageCrop',imgCrop)
        cv2.imshow('ImageWhite',imgWhite)

        key = cv2.waitKey(1)
        if key == ord('s'):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(f'Image saved: {counter}')

    else:
        # No hand detected, show just webcam
        cv2.imshow('Image', img)

    if not success:
        print("Failed to grab frame")
        break

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to exit the camera
        break

cap.release()
cv2.destroyAllWindows()

