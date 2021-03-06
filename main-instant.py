import cv2 as cv

import numpy as np

huver=str()
huclass=str()

#This function checks whether the input image is human or non human

def human_verification(x):
    
    
    img = cv.imread(x)
    gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)   
    haar_cascade = cv.CascadeClassifier('-------/haarcascade_frontalface_default.xml') #Reads the haarcascade file and store in the variable

    faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    hvr=""
    
    if(len(faces_rect)>0):
            hvr='Human'
            
            cv.putText(img,"Human",(40,40),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)

    else:
        hvr='Non Human'   
        cv.putText(img,"Non Human",(40,40),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)

    cv.imshow('Detected',img)
    print("Press any key to continue...")
    cv.waitKey(0)
    cv.destroyAllWindows()
    return hvr 
        
def human_classification(y):
    print("Please wait,it takes a while to read the yaml file.")
  
    haar_cascade =cv.CascadeClassifier(r'D:/---haarcascade_frontalface_default.xml')

    people =['Indian','Non-Indian']
    features =np.load('features.npy',allow_pickle=True)
    labels =np.load('labels.npy')
    
    face_recognizer=cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_trained.yml')


    img =cv.imread(y)
    gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)    
    m = -1.0
    i = None

    faces_rect =haar_cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces_rect:
            faces_roi=gray[y:y+h,x:x+w]
            
            label,confidence =face_recognizer.predict(faces_roi)
            if i is None:
                i=label
                if (confidence>m):
                    i=label
                    m=confidence
                    
    hcs=people[i]               


    cv.putText(img,str(people[i]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
    
    cv.imshow('Image Result',img)
    print("Press any key to continue...")
    cv.waitKey(0)        
    return hcs


print("This program detects a human in the image and if there is one then it tells wheteher he/she is Indian or Non Indian")

####### This module will capture real time photo #######################
 
import cv2 

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        print(check)            #prints true as long as the webcam is running
        print(frame)            #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='captured_img.jpg', img=frame)
            webcam.release()
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            img_ = cv2.imread('captured_img.jpg', cv2.IMREAD_ANYCOLOR)
            print("Image saved!")
        
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
    
path=r"D:/----/captured_img.jpg"  
#Path of the instantly-captured image

huver=human_verification(path)
               
if (huver=='Human'):
        
    huclass=human_classification(path)
        
    print(huclass)
            

else:
    print("No Human detected in the image")

