# Five-Fold-Cross-Validation-of-CNN-Model-using-Image-Sentiment-Prediction
Five-Fold-Cross-Validation-of-CNN-Model-using-Image-Sentiment-Prediction
This is the CNN Model for which five fold cross validation is demonstrated on image sentiment prediction and final accuracy is given.


*******************************************************************************************************
This is work is on Five Fold Cross Validation of CNN Model using Image Sentiment Prediction
*******************************************************************************************************

Package Version
python 3.6.8
numpy  1.16.1  
skimage 0.16.2
pandas 0.24.1
keras 2.2.4                  

The CNN model is built with 7 convolutoinal layers. This model is trained on the images to perform the sentiment prediction.
The images are read from the given folder and corresponding ground truth labels are read into a data frame.
From the input images a image set suitable for CNN model is prepared.

This code provides the demo for five fold cross validation. The image data is split using stratified folds and five folds are created.
The folds are created by preserving the percentage of samples for each class. Four folds are used for training and one for testing.
The model is evaluated on the one fold and accuracy is computed. This procedure is continued five time by chaning testing fold everytime.
The cross validation accuracy is computed as taking the average of total score.

Please set the imPath to the folder consisting of images seperated by two backslash

For example: c:\\images\\

A sample of images are kept here, please see the below link for complete image set.
http://www.ee.columbia.edu/ln/dvmm/vso/download/twitter_dataset.html

## Further Projects and Contact
## www.researchreader.com

## https://medium.com/@dr.siddhaling

## Dr. Siddhaling Urolagin,
dr.siddhaling@gmail.com

