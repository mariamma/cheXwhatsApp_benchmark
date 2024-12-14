## Basic libraries
import numpy as np
import pandas as pd
import os

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
import utils
import transforms as T
import os

import argparse
from lung_data import LungData

classes = [-1, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
## Parent path to data
path = '/data/mariammaa/vindr/dataset/'
imageFolders = ['png_train/', 'png_test/']

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


def load_model(model, folder, net_basename, name):
    # state = torch.load(f"{folder}{net_basename}_{name}_model.pkl")
    state = torch.load(f"{folder}{net_basename}")
    model.load_state_dict(state['model_rep'])
    return model


def ols_score(fname, image_size, lung_segment_path, heatmap_hr,heatmap_lr, row, grad_type):
    lung_region = np.load(os.path.join(lung_segment_path,fname.replace(".png",".npy")))
    lung_region = resize(lung_region, (image_size, image_size))
        
    heatmap_hr = heatmap_hr.cpu().detach().numpy()
    heatmap_lr = heatmap_lr.cpu().detach().numpy()        
    if np.sum(lung_region) > 0:
        for threshold in [0, 0.00005, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]:
            # print("Numerator :", np.sum((heatmap_hr>threshold)), np.sum((lung_region>0)) , np.sum((heatmap_hr>threshold)*(lung_region>0)))
            # print("Denominator :", np.sum(heatmap_hr>threshold)+0.00000001)
            dice_hr = (np.sum((heatmap_hr>threshold)*(lung_region>0))/(np.sum(heatmap_hr>threshold)+0.00000001))
            dice_lr = (np.sum((heatmap_lr>threshold)*(lung_region>0))/(np.sum(heatmap_lr>threshold)+0.00000001))
            row[grad_type+'HR_Score_' + str(threshold)] = dice_hr
            row[grad_type+'LR_Score_' + str(threshold)] = dice_lr
            print("Threshold:{}, hr:{}, lr:{}".format(threshold, dice_hr, dice_lr))
    return row      


def create_png_df(boxLabel, ):
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


def init(bboxfile, tasks):
   
    ## Image data
    # dataDF = pd.read_csv(path + dataCSV)
    ## Box data
    boxDF_orig = pd.read_csv(bboxfile)

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
    print(boxDF.head())

    boxDF = boxDF[boxDF['Target'].isin(tasks)]
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


def test(args):
     
    height = 256
    width = 256
    wh_pathDir = "/data/mariammaa/vindr/dataset/wh_proper_train/"
    lung_segment_path = "/data/mariammaa/vindr/dataset/png_test_lung_segment/"
    target_dir = "/data/mariammaa/vindr/saved_results/"
    dirname = "/home/mariammaa/cheXwhatsApp_benchmark/Training/Object_Detection/VinBig/RCNN/"
    # train_csv = "vindr_train.csv"
    val_csv = "vindr_val.csv"

    print(args.tasks)
    tasks = [int(x) for x in args.tasks]
    print("tasks : ", tasks)

    xBox, yBox, wBox, hBox, boxLabel = init(os.path.join(dirname, val_csv), tasks)
    pngImages = create_png_df(boxLabel)
    dataTest = LungData(height, width, xBox, yBox, wBox, hBox, boxLabel, 
        pngImages, classes, global_transformer(), wpPath=wh_pathDir)

    ## Split the dataset into train and test sets
    torch.manual_seed(1)

    dataTestLoader = torch.utils.data.DataLoader(dataTest, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    print("Dataloader length :", len(dataTestLoader))

    ## Determine if we can use a GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ## Initialize model
    model = detectionModel(args.num_classes)

    model = load_model(model, args.model_folder, args.net_basename, args.model_name)
    model.to(device)

    model.eval()
    
    for fname, orig_img, wp_img, target in dataTestLoader:
        print(fname)
    # df = compute_rho(args, model, dataTestLoader, lung_segment_path, device=device)
    # filename = args.net_basename + ".csv"  
    # df.to_csv(os.path.join(target_dir, filename), index=False)                              





def main():
    parser = argparse.ArgumentParser(description='Multi-task Learning Trainer')
    parser.add_argument('--net_basename', type=str, default='vindr_obj_det_5', help='model name')
    parser.add_argument('--model_name', type=str, default='best', help='model name')
    parser.add_argument('--model_folder', type=str, default='/data/mariammaa/vindr/saved_models/', help='model folder')
    parser.add_argument('--num_classes', type=int, default=16, help='Batch size')
    parser.add_argument('--tasks', nargs = '+' , default=[1,2], help='tasks to be trained')
    args = parser.parse_args()
    test(args)


main()