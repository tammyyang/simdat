import numpy as np
import cv2

imgpath = '/home/tammy/www/test/IMAG2646.jpg'
face_cascade = cv2.CascadeClassifier('/home/tammy/SOURCES/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

img = cv2.imread(imgpath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

# cv2.imwrite('output.jpg', img)

