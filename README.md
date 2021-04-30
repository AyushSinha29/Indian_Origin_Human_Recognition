# Indian_Origin_Human_Recognition

The main motive of the model is to detect whether the input image of human belongs to Indian Origin or not.
This repository contains a test file , train file and a webscrapper along with the face haarcascade that was used.

Training the model:
To train the model just put in the path of the train folder and create two folders : Indian and Non - Indian, containing the respective images. And simply run the train file in the same directory. The train file will read the images from the corresponding folders and will create three files: features.npy , labels.npy and face_trained (yml) file. These three files saves the features of image and uses them while testing.

Testing the model :
The first part of the test file (human_verification) will verify that the input image is human or non-human and gradually the next part (human_classification) will classify them as Indian or Non Indian.For this , the model will read the three files created in the training stage.
