import cv2
import numpy as np

detect = cv2.CascadeClassifier ('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture ('khafid.jpg')
check, img = cam.read()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = detect.detectMultiScale(gray, 1.2, 5)

for (x, y, w, h) in face:
    cv2.rectangle (img, (x,y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Face Detect', img)
cv2.waitKey(600000)

cam.release()
cv2.destroyWindow()
