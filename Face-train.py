import os
import cv2
import numpy as np
from PIL import Image
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'/Users/daadhiwalebaba/Downloads/Dataset' )
faceCascade = cv2.CascadeClassifier('/Users/daadhiwalebaba/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
#LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

 
current_id = 0
label_ids = {}
y_labels = []
x_train = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg")or file.endswith("jpeg")or file.endswith("png"):
            path  = os.path.join(root,file)
            label = os.path.basename(root).replace(" ", "-").lower()
            #print(path,label)
            if not label in label_ids:
            
                label_ids[label] = current_id
                current_id += 1
                
            id_ = label_ids[label]
            print(label_ids)
            #y_labels.append(label)
            #x_train.append(path)
            
            pil_image = Image.open(path).convert("L")
            size = (300,300)
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            image_array = np.array(pil_image,"uint8")
            #print(image_array)
            faces = faceCascade.detectMultiScale(image_array,scaleFactor=1.2,minNeighbors=16,minSize=(30, 30)) 
            for (x, y, w, h) in faces:
                 roi = image_array[y:y+h,x:x+w]
                 x_train.append(roi)
                 y_labels.append(id_)
    
#print(y_labels)
#print(x_train)
with open("labels.pickle",'wb') as f:
     pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("recognizer/trainingData.yml")
