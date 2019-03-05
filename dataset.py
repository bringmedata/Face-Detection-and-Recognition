import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('/Users/daadhiwalebaba/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 4928)
cap.set(4, 3264)
id = input('Enter User ID')
sampleN=0;

while 1:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.2, 5)
	for (x,y,w,h) in faces:
		sampleN=sampleN+1;
		cv2.imwrite('/Users/daadhiwalebaba/Downloads/Dataset/1.0/1.'+str(sampleN)+ ".jpg", gray[y:y+h, x:x+w])
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
		cv2.waitKey(100)
	cv2.imshow('img',img)
	cv2.waitKey(1)
	if sampleN > 14:
		break
cap.release()
cv2.destroyAllWindows()	
