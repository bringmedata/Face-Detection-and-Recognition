
import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

face_cascade = cv2.CascadeClassifier('/Users/daadhiwalebaba/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/daadhiwalebaba/opencv/data/haarcascades_cuda/haarcascade_eye.xml.xml')

rec = cv2.face.LBPHFaceRecognizer_create();
rec.read('/Users/daadhiwalebaba/recognizer//trainingData.yml')

id = 0

font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 10)
	print('Number of Faces Found:',len(faces))
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		id, conf = rec.predict(gray[y:y+h, x:x+w])
		print("CONFIDENCE is:",conf)
		if (conf>50):
			if (id==0):
				id = 'Abhishek Rana'
			elif (id==1):
				id = 'Nikunj Sharma'
			elif (id==2):
				id = 'Shreya Singh'
		elif (conf<50):
			id = 'Unkown ID'
		cv2.putText(frame, str(id), (x,y+h), font, 0.5, (255,255,255), 1, cv2.LINE_AA);
	cv2.imshow('Webcam Facial Recognition',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
