import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
import pandas as pd
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Activation,  Flatten
from keras.optimizers import sgd
from sklearn.model_selection import StratifiedKFold

imPath='\images'
grndTruthPath='groundTruthLabel.txt'

#Required dimension of the input images
imageDim=(256,256)
inputShape = (imageDim[0],imageDim[1],3)
#number of classes
numOfClasses = 2

#collect all the directors and files present in the path
dirs=os.listdir(imPath)
#Create a data frame consisting of grpund truth values
clNames=['Image','Sentiment']
clsLabels=pd.read_csv(grndTruthPath,names=clNames,delimiter='\t')
clsLabels.set_index('Image',inplace=True)

#read images from given paths and prepare images set
def createImagesSet(allImagesFoldrPath,imageDim,clsLabels):
    x_imageSet=np.empty((len(allImagesFoldrPath),imageDim[0],imageDim[1],3))
    imDbDict={}
    y_Set=np.empty((len(allImagesFoldrPath),1))
    for im in range(len(allImagesFoldrPath)):
        readImage=imread(allImagesFoldrPath[im])
        print(allImagesFoldrPath[im])
        imName=allImagesFoldrPath[im].split('\\')[-1]
        actualClass=clsLabels.loc[imName,'Sentiment']
        
        if (actualClass=='positive'):
            y_Set[im]=1
        else:
            y_Set[im]=0
            
        if (len(readImage.shape)>=3):
            if readImage.shape[2]>3:
                readImage=readImage[:,:,:3]            
        else:
            print(im,readImage.shape)
            readImage=gray2rgb(readImage)            
        readImage=resize(readImage,imageDim)
        x_imageSet[im]=readImage
        imDbDict[allImagesFoldrPath[im]]=(x_imageSet[im],y_Set[im])
    return imDbDict

#collect image names from the path list and check if its name is present in the groundTruth or not
def collectImNames(entireDb):
    imNmPresentInGrndTrth=[]
    imPathNotPresentInGrndTrth=[]
    for imPath in range(len(entireDb)):
        imNm=entireDb[imPath].split('\\')[-1]
        if imNm in clsLabels.index:
            imNmPresentInGrndTrth.append(imNm)
        else:
            imPathNotPresentInGrndTrth.append(entireDb[imPath])
    return imNmPresentInGrndTrth,imPathNotPresentInGrndTrth

#load the train and test images into two arrays of images.Convert to float type and normalize by dividing 255
def loadPrepareData(allImagesTrainPath,allImagesTestPath,imDbDict):
    x_trainImSet=np.empty((len(allImagesTrainPath),imageDim[0],imageDim[1],3))
    x_testImSet=np.empty((len(allImagesTestPath),imageDim[0],imageDim[1],3))
    y_trainSet=np.zeros(len(allImagesTrainPath))
    y_testSet=np.zeros(len(allImagesTestPath))
    for trnPi in range(len(allImagesTrainPath)):
        (x_trainImSet[trnPi],y_trainSet[trnPi])=imDbDict[allImagesTrainPath[trnPi]]
    
    for testPi in range(len(allImagesTestPath)):
        (x_testImSet[testPi],y_testSet[testPi])=imDbDict[allImagesTestPath[testPi]]
        
    x_trainImSet= x_trainImSet.astype('float32')
    x_testImSet= x_testImSet.astype('float32')
    x_trainImSet /= 255.0
    x_testImSet /= 255.0
    y_trainSetBinary=y_trainSet
    y_testSetBinary=y_testSet
    # convert class vectors to matrices as binary
    y_trainSet= keras.utils.to_categorical(y_trainSet, numOfClasses)
    y_testSet= keras.utils.to_categorical(y_testSet, numOfClasses)
    
    print('Number of train samples in Dataset: ', x_trainImSet.shape[0])
    print('Number of test samples in Dataset: ', y_testSet.shape[0])
    
    return (x_trainImSet,y_trainSet,y_trainSetBinary), (x_testImSet,y_testSet,y_testSetBinary)

# Build Convolutional Neural Network Model
def buildModel():
    model = Sequential()
    
    # Convolution layer 1
    model.add(Conv2D(filters=96, kernel_size=(11,11), padding='same',
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', 
                     input_shape=inputShape))
    model.add(Activation('relu'))
    # Convolution layer 2
    model.add(Conv2D(192, kernel_size=(5,5), padding='same', 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Convolution layer 3
    model.add(Conv2D(192, kernel_size=(5,5), padding='same', 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Convolution layer 4
    model.add(Conv2D(192, kernel_size=(5,5), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Convolution layer 5
    model.add(Conv2D(96, kernel_size=(3,3), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, kernel_size=(3,3), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(48, kernel_size=(1,1), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Flatten())
    
    # Dense layer with softmax
    model.add(Dense(numOfClasses, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('softmax'))
    
    return model

#Data frame to store performance measurements
performanceTb=pd.DataFrame(columns=['FoldNum','TrnOrTest','TN', 'FP', 'FN', 'TP'])
batchSz = 10
epochs = 2
seed = 11

#collect all the path of images
allImsPaths=[(imPath+di) for di in dirs if('txt' not in di)]
#remove images not present in ground truth table
imNmPresentInGrndTrth,imPathNotPresentInGrndTrth=collectImNames(allImsPaths)
labels=list(clsLabels.loc[imNmPresentInGrndTrth,'Sentiment'])
for rPath in imPathNotPresentInGrndTrth:
    allImsPaths.remove(rPath)

#create an image data set
imDbDict=createImagesSet(allImsPaths,imageDim,clsLabels)

#create 5 statified folds
statifiedFolds = StratifiedKFold(n_splits=5,shuffle=True, random_state=seed)

#initialize and create the CNN model
model = buildModel()
#Set the optimizer for CNN model
optimizer = sgd(0.01, 0.9, 0.0005, nesterov=True)
#compile the CNN model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
Total_score=0
#perform the Cross validation on CNN model
foldi=0
for trainIndx, testIndx in statifiedFolds.split(allImsPaths, labels):
    print("trainIndx: %s testIndx: %s" % (len(trainIndx), len(testIndx)))    
    trainSetImagesPath=[allImsPaths[indx] for indx in trainIndx]
    testSetImagesPath=[allImsPaths[indx] for indx in testIndx]
    (x_trainImSet,y_trainSet,y_trainSetBinary), (x_testImSet,y_testSet,y_testSetBinary)=loadPrepareData(trainSetImagesPath,testSetImagesPath,imDbDict)
    model.fit(x_trainImSet, y_trainSet,batch_size=batchSz,epochs=epochs,validation_data=(x_testImSet, y_testSet),shuffle=True)
    score = model.evaluate(x_testImSet, y_testSet)
    Total_score+=score
    foldi=foldi+1
    
print("Cross validation Accuracy=",Total_score)