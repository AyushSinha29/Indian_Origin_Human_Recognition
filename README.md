# Indian_Origin_Human_Recognition

The main motive of the model is to detect whether the input image of human belongs to Indian Origin or not.
This repository contains a train, Human classification, test(Indian or non Indian verification) and main file and also the face haarcascade file used in the model.

The dataset used for the project can be found at- https://www.kaggle.com/sinhayush29/indian-people (for Indian Images) and https://www.kaggle.com/sinhayush29/non-indian-people (for Non Indian Images)

Training the model: (USE OF train.py)
To train the model just put in the path of the train folder and create two folders : Indian and Non - Indian, containing the respective images. And simply run the train file in the same directory. The train file will read the images from the corresponding folders and will create three files: features.npy , labels.npy and face_trained (yml) file. These three files saves the features of image and uses them while testing.

Testing the model : (USE OF main.py)
The first part of the main file (human_verification) will verify that the input image is human or non-human and gradually the next part (human_classification) will classify them as Indian or Non Indian.For this , the model will read the three files created in the training stage.

Testing the model instantly: (USE OF main-instant.py)
This piece of code will capture the image of user instantly and will process the image through all the processess similar to that of main.py, as mentioned above.
Just click on 's' and see the magic happening!
