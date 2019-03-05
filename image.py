import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('/Users/daadhiwalebaba/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/daadhiwalebaba/opencv/data/haarcascades_cuda/haarcascade_eye.xml.xml')

img = cv2.imread('/Users/daadhiwalebaba/Downloads/hall.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 3)

print('Number Of Faces Found:',len(faces))

for (x,y,w,h) in faces:
	img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),1)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

