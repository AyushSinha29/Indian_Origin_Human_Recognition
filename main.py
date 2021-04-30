import cv2 as cv

import numpy as np

huver=str()
huclass=str()

#This function checks whether the input image is human or non human
def human_verification(x):
    
    
    img = cv.imread(x)
    
    
    gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
   

    haar_cascade = cv.CascadeClassifier('C:/Users/KIIT/Desktop/Human Image Proj/Sample/haarcascade_frontalface_default.xml') #Reads the haarcascade file and store in the variable

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
  
    haar_cascade =cv.CascadeClassifier('C:\\Users\\KIIT\\Desktop\\Human Image Proj\\haarcascade_frontalface_default.xml')

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


print("This program detects a human in the image and if there is one then it tells wheteher he/she is Indian or Non Indian and detects the skintone if Indian")


    
path=input("Enter path of image:")
    
huver=human_verification(path)
    
    
        
if (huver=='Human'):
        
    huclass=human_classification(path)
        
    print(huclass)
            

else:
    print("No Human detected in the image")

############################################   SKIN TONE DETECTION    #######################################################


#This module detects the face in the input image

if (huclass=='Indian'):
    img = cv.imread(path) #Read the input image
  
# Convert into grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier('C:/Users/KIIT/Downloads/SkinColor/haarcascade_frontalface_default.xml')
  
# Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    for (x, y, w, h) in faces:
    
      
     faces = img[y:y + h, x:x + w]
    
     
     cv.imwrite('face.jpg', faces)
    
  

    cv.waitKey()

#.........................................................................................

#This module detects the skin portion in the input image.

    import numpy as np

    minRange = np.array([0,133,77],np.uint8) #for min skin color Range
    maxRange = np.array([235,173,127],np.uint8) #for maximum skin color Range
    image = cv.imread(r"C:\Users\KIIT\Desktop\Human Image Proj\dataset_final\dataset\face.jpg")

    # change our image bgr to ycr using cvtcolor() method 
    YCRimage = cv.cvtColor(image,cv.COLOR_BGR2YCR_CB)

    # apply min or max range on skin area in our image
    skinArea = cv.inRange(YCRimage,minRange,maxRange)
    detectedSkin = cv.bitwise_and(image, image, mask = skinArea)

    cv.imwrite(r"C:\Users\KIIT\Desktop\Human Image Proj\dataset_final\dataset\detectedImage.png", 
               np.hstack([detectedSkin]))

#..................................................................................................................

#This module collects the color and checks with the provided range and classifies accordingly.

    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import numpy as np
    
    from collections import Counter
    from skimage.color import rgb2lab, deltaE_cie76
    from PIL import Image
    from numpy import asarray
    
    image = cv.imread(r'C:\Users\KIIT\Desktop\Human Image Proj\dataset_final\dataset\detectedImage.png')

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    resized_image = cv.resize(image, (1200, 600))


    def RGB2HEX(color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


    def get_image(image_path):
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return image


    # convert image to numpy array
    data = asarray(image)


    # create Pillow image
    image2 = Image.fromarray(data)


    modified_image = cv.resize(image, (600, 400), interpolation = cv.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)


    clf = KMeans(n_clusters = 2)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]   
    

    c1=hex_colors[0]
    c2=hex_colors[1]

    d1=c1.lstrip('#')
    d2=c2.lstrip('#')

    res1 = int(d1, 16)
    res2 = int(d2, 16)

    if 8421504>res1>0: #to ignore the darker shade of the background
        fc=res2
        print("The detected skin tone is:",res2," with hex color code as",c2)
    
    else:
        fc=res1
        print("The detected skin tone is:",res1,"(bg color) with hex color code as",c1)  

    
    print("The detected skintone is:")
    font = cv.FONT_HERSHEY_TRIPLEX

    if 16777215>fc>12619362: #fair range
        print("Fair")
     
        cv.putText(img,'Indian, Skin Tone: FAIR',(10,50), font, 1,(0,255,0),2)
        cv.imshow("Result",img)
        cv.waitKey(0)

    
    elif 12619362>fc>10300000:    #mild range
        print("Mild")
        
        cv.putText(img,'Indian, Skin Tone: MILD',(10,50), font, 0.5,(0,255,0),2)
        cv.imshow("Result",img)
        cv.waitKey(0)
    
    else:
        print("Dark")    
        cv.putText(img,'Indian, Skin Tone: DARK',(10,50), font, 0.5,(0,255,0),2)
        cv.imshow("Result",img)
        cv.waitKey(0)
        
else:
    original=cv.imread(path)
    
    font = cv.FONT_HERSHEY_TRIPLEX
    cv.putText(original,'Skin tone not detected as person is Non-Indian',(10,50), font, 0.5,(0,255,0),2)
    cv.imshow("Result",original)
    cv.waitKey(0)
    

  

 

  
    
















