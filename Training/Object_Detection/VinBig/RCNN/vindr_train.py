## Basic libraries
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from sklearn.model_selection import train_test_split
# import pydicom
from skimage.transform import resize

## Torchvision libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torch.utils.tensorboard import SummaryWriter

## Image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


## Helper libraries
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import math
import imageio
import os
import shutil
from PIL import Image,ExifTags

from lung_data import LungData

 ## Parent path to data
path = '/data6/rajivporana_scratch/vindr_data/'

boxCSV = 'train.csv'
# dataCSV = 'stage_2_detailed_class_info.csv'

imageFolders = ['png_train/', 'png_test/']

## First class is background
classes = [-1, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

## Create the base model to be trained
def detectionModel(numClasses):
    ## Load a model pretrained resnet model to speed training time
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    ## Get number of input features for the classifier
    inFeatures = model.roi_heads.box_predictor.cls_score.in_features
    ## Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(inFeatures, numClasses) 

    return model

def global_transformer():
    return transforms.Compose([transforms.ToTensor()])

## Data transformations during training to reduce overfit
def createTransform(train):
    if train:
        return A.Compose([A.HorizontalFlip(0.5), ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



def init():
   
    ## Image data
    # dataDF = pd.read_csv(path + dataCSV)
    ## Box data
    boxDF_orig = pd.read_csv(path + boxCSV)

    boxDF = pd.DataFrame()
    for index, row in boxDF_orig.iterrows():
        row_new = {}
        row_new['patientId'] = row['image_id']
        row_new['x'] = row['x_min']
        row_new['y'] = row['y_min']
        row_new['width'] = row['x_max'] - row['x_min']
        row_new['height'] = row['y_max'] - row['y_min']
        row_new['Target'] = row['class_id']
        boxDF = pd.concat([boxDF, pd.DataFrame([row_new])], ignore_index=True)
    # print(boxDF.head())

    dataGlob = glob('/data6/rajivporana_scratch/vindr_data/png_train/*.png')


    ## Dictionary of image to path
    imagePaths = {os.path.basename(x)[:-4]: x for x in dataGlob}
    ## Full image path data table
    # dataPath = dataDF['patientId'].map(imagePaths.get)

    ## Full path to images with bounding boxes
    boxPaths= boxDF['patientId'].map(imagePaths.get)

    ## Isolate images with bounding boxes
    # boxImages = pd.merge(left = boxDF, right = dataDF, left_on = 'patientId', right_on = 'patientId', how = 'inner')
    # boxImages.dropna(axis = 0, inplace = True)
    boxDF.dropna(axis = 0, inplace = True)

    ## Bounding box and label data groupings by image. Using boxDF due to inner join duplication
    xBox = boxDF.groupby('patientId')['x'].apply(np.array).reset_index()['x'].values
    yBox = boxDF.groupby('patientId')['y'].apply(np.array).reset_index()['y'].values
    wBox = boxDF.groupby('patientId')['width'].apply(np.array).reset_index()['width'].values
    hBox = boxDF.groupby('patientId')['height'].apply(np.array).reset_index()['height'].values
    ## Group the finding labels together for the varying bounding boxes in each image
    boxLabel = boxDF.groupby('patientId')['Target'].apply(np.array).reset_index()

    print(xBox)

    print(boxLabel.head())

    boxLabel['paths'] = os.path.join(path,imageFolders[0]) + boxLabel['patientId'] + ".png"

    print("Number of images: ", len(boxLabel))
    print(boxLabel.head())
    return xBox, yBox, wBox, hBox, boxLabel



def create_png_df(boxLabel):
    imageDict = {'Directory':[], 'ID':[]}
    
    for i in range(len(boxLabel)):
        try:
            imageName = boxLabel.iloc[i]['patientId']
            jpgName = imageName + ('.png')         
            train_dir = os.path.join(path,imageFolders[0])   
            imageDict['Directory'].append(os.path.join(train_dir, jpgName))
            imageDict['ID'].append(imageName)
        except Exception as e:
            print(i,e)

    return pd.DataFrame(imageDict)




def train():
    xBox, yBox, wBox, hBox, boxLabel = init()
    pngImages = create_png_df(boxLabel)


    dataTrain = LungData(256, 256, xBox, yBox, wBox, hBox, boxLabel, 
        pngImages, classes, global_transformer())
    dataTest = LungData(256, 256, xBox, yBox, wBox, hBox, boxLabel, 
        pngImages, classes, global_transformer())

    ## Split the dataset into train and test sets
    torch.manual_seed(1)
    # indices = torch.randperm(len(dataTrain)).tolist()
    indices = torch.arange(len(dataTrain)).tolist()
    dataSplit = 0.2
    testSize = int(len(dataTrain)*dataSplit)
    dataTrain = torch.utils.data.Subset(dataTrain, indices[:-testSize])
    dataTest = torch.utils.data.Subset(dataTest, indices[-testSize:])

    ## Define training and validation data loaders
    dataTrainLoader = torch.utils.data.DataLoader(dataTrain, batch_size=10, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    dataTestLoader = torch.utils.data.DataLoader(dataTest, batch_size=10, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    ## Determine if we can use a GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = "cuda:7"

    ## Initialize model
    numClasses = 16
    model = detectionModel(numClasses)

    # model.double()
    model.to(device)

    ## Construct an optimizer and learning rate
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0055, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
    lrScheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    ## Choose number of training epochs
    numEpochs = 5

    ## Tensorboard writer
    # writer = SummaryWriter()
    print("Dataloader length :", len(dataTrainLoader))
    ## Train one epoch at a time for numEpochs
    for epoch in range(numEpochs):
        train_one_epoch(model, optimizer, dataTrainLoader, device, epoch, print_freq=10)
        ## Update the learning rate
        lrScheduler.step()
        ## Evaluate on the test dataset
        evaluate(model, dataTestLoader, device=device)
        
        torch.save(model.state_dict(), f"/data6/rajivporana_scratch/models/vindr_models/vindr_obj_det_{epoch+1}")


def main():
    train()


main()