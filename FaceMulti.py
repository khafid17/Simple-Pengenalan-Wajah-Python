#library OpenCv
import cv2,glob

gimg = glob.glob('jalan.jpg')

#gunkaan face detection frontal
detect = cv2.CascadeClassifier ('haarcascade_frontalface_default.xml')

for timg in gimg:
    img = cv2.imread(timg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #koordinat
    face = detect.detectMultiScale(gray, 1.15,5)

    for (x,y,w,h) in face:
        cv2.rectangle (img, (x,y), (x+w, y+h), (0, 225, 0), 2)

    cv2.imshow('Face Detect', img)
    cv2.waitKey(2000)
    cv2.destroyWindow()
