import cv2 as cv
import matplotlib.pyplot as plt 

for i in range(1,174):
    img = cv.imread('C:/Users/KIIT/Desktop/Human Image Proj/Sample/input ('+str(i)+').jpg')
    plt.subplot(2,2,1)
    #cv.imshow(str(i),img)

    gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #cv.imshow('Gray',gray)

    haar_cascade = cv.CascadeClassifier('C:/Users/KIIT/Desktop/Human Image Proj/Sample/haarcascade_frontalface_default.xml') #Reads the haarcascade file and store in the variable

    faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5)

    print(f'Number of faces detected = {len(faces_rect)}')

    for (x,y,w,h) in faces_rect:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

    plt.subplot(2,2,2)

    if(len(faces_rect)>0):
        cv.putText(img,"Human",(40,40),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)

    else:
        cv.putText(img,"Non Human",(40,40),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)

    cv.imshow('Detected',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
