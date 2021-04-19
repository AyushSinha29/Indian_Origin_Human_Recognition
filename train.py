import os
import cv2
import numpy as np

people =['Indian', 'Non-Indian']
DIR = r'C:\\Users\\KIIT\Desktop\\Human Image Proj\\dataset\\train'

#for i in os.listdir(r'F:\Opencv\Photos 2'):
    #p.append(i)
#print(p)

haar_cascade =cv2.CascadeClassifier('C:\\Users\\KIIT\\Desktop\\Human Image Proj\\dataset\\train\\haarcascade_frontalface_default.xml')

features =[] #  Image arrays
labels =[] #Names of the people

def create_train():
    for person in people:
        path =os.path.join(DIR,person) #Finding path to a folder
        label =people.index(person) #finding the index of the person inpeople list

        for img in os.listdir(path): #looping in the person folder
            img_path =os.path.join(path,img) #finding the image path in the folder
            
            img_array =cv2.imread(img_path) #reading the image of the person
            if img_array is None:
                continue
            gray =cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)

            faces_rect =haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

            for (x,y,w,h) in faces_rect:
                faces_roi =gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)


create_train()
print('-------------Training done -----------------')

features =np.array(features,dtype='object') #converting from lists to numpy arrays
labels =np.array(labels)

face_recognizer =cv2.face.LBPHFaceRecognizer_create() #Inbuilt Face recognizer

#Train the recognizer on features list and labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)