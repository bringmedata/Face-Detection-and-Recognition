import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT

face_cascade = cv2.CascadeClassifier('/Users/daadhiwalebaba/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/daadhiwalebaba/opencv/data/haarcascades_cuda/haarcascade_eye.xml.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    print('Number Of Faces Found:',len(faces))
    # Display the resulting frame
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
         
         
         

    cv2.imshow('Webcam Facial Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
