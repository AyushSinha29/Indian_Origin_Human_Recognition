import os
import cv2
import numpy as np

people =['Indian', 'Non-Indian']
DIR = r'D:/----/train' #path of the training folder

haar_cascade =cv2.CascadeClassifier('D:/-----/haarcascade_frontalface_default.xml') #path of haarcascade file

features =[]    #Image arrays
labels =[]      #Names of the people

print("Please wait, it will take a while to train the model....")

def create_train():
    for person in people:
        
        path =os.path.join(DIR,person)          #finding path to a folder
        label =people.index(person)             #finding the index of the person in people list

        for img in os.listdir(path):            #looping in the person folder
            img_path =os.path.join(path,img)    #finding the image path in the folder
            
            img_array =cv2.imread(img_path)     #reading the image of the person
        
            if img_array is None:
                continue
            gray =cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)

            faces_rect =haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

            for (x,y,w,h) in faces_rect:
                faces_roi =gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('------------- Training done -----------------')
print('------------  Head on to main.py ------------')

features =np.array(features,dtype='object')             #converting from lists to numpy arrays
labels =np.array(labels)

face_recognizer =cv2.face.LBPHFaceRecognizer_create()   #Inbuilt Face recognizer

#Train the recognizer on features list and labels list
face_recognizer.train(features,labels)

#The trained features are now saved as yaml file 
face_recognizer.save('face_trained.yml')

np.save('features.npy',features)
np.save('labels.npy',labels)
