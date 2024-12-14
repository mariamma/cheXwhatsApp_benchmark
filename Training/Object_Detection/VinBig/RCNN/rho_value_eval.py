## Basic libraries
import numpy as np
import pandas as pd
import os

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
import os

from lung_images import LungImages
import argparse

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


def load_model(model, folder, net_basename, name):
    # state = torch.load(f"{folder}{net_basename}_{name}_model.pkl")
    state = torch.load(f"{folder}{net_basename}")
    model.load_state_dict(state)
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


def compute_rho(args, model, dataTestLoader, lung_segment_path, device):
    model.eval()
    image_size = 256
    df = pd.DataFrame()

    for fname, orig_img, wp_img in dataTestLoader:
        fname = fname[0]
        orig_img = [img.to(device) for img in orig_img]
        for img in orig_img: img.requires_grad_()
        res_orig = model(orig_img)

        wp_images = [img.to(device) for img in wp_img]
        for img in wp_images: img.requires_grad_()
        res_wp = model(wp_images)

        res_orig = res_orig[0]
        res_orig_bboxes = res_orig['boxes']
        res_orig_labels = res_orig['labels'].cpu().detach().numpy()
        res_wp = res_wp[0]
        res_wp_bboxes = res_wp['boxes']
        res_wp_labels = res_wp['labels'].cpu().detach().numpy()

        row={}
        for i in range(15):
            if i in res_orig_labels and i in res_wp_labels:
                for j in range(i+1, 15):
                    if j in res_orig_labels and j in res_wp_labels:
                        orig_label_idx = np.where(res_orig_labels == i)
                        wp_label_idx = np.where(res_wp_labels == j)

                        orig_bbox = res_orig_bboxes[orig_label_idx]
                        wp_bbox = res_wp_bboxes[wp_label_idx]        

                        det_sal_robustness = torch.autograd.grad(orig_bbox.sum(), orig_img, retain_graph=True)[0]
                        print("Det saliency:", det_sal_robustness.shape)

                        seg_sal_robustness = torch.autograd.grad(wp_bbox.sum(), wp_images, retain_graph=True)[0]
                        print("Seg saliency:", seg_sal_robustness.shape)

                        grad_type = str(i) + "_" + str(j)
                        ols_score(fname, image_size, lung_segment_path, det_sal_robustness, seg_sal_robustness, row, grad_type)

                        det_sal_robustness = torch.flatten(det_sal_robustness)
                        seg_sal_robustness = torch.flatten(seg_sal_robustness)
                        det_correlation = torch.dot(det_sal_robustness, seg_sal_robustness) / (torch.norm(det_sal_robustness) * torch.norm(seg_sal_robustness))
                        print("Correlation : {},{}={}".format(i,j,det_correlation))
                        row[grad_type+'_corr'] = det_correlation
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)  
    return df


def test(args):
     
    height = 256
    width = 256
    pathDir = "/data/mariammaa/vindr/dataset/png_test"
    wh_pathDir = "/data/mariammaa/vindr/dataset/wh_proper_test/"
    lung_segment_path = "/data/mariammaa/vindr/dataset/png_test_lung_segment/"
    target_dir = "/data/mariammaa/vindr/saved_results/"
    dataTest = LungImages(pathDir, wh_pathDir, height, width, 
            classes, transforms =  global_transformer())

    ## Split the dataset into train and test sets
    torch.manual_seed(1)

    dataTestLoader = torch.utils.data.DataLoader(dataTest, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    print("Dataloader length :", len(dataTestLoader))

    ## Determine if we can use a GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ## Initialize model
    numClasses = 16
    model = detectionModel(numClasses)

    model = load_model(model, args.model_folder, args.net_basename, args.model_name)
    model.to(device)

    df = compute_rho(args, model, dataTestLoader, lung_segment_path, device=device)
    filename = args.net_basename + ".csv"  
    df.to_csv(os.path.join(target_dir, filename), index=False)                              


def main():
    parser = argparse.ArgumentParser(description='Multi-task Learning Trainer')
    parser.add_argument('--net_basename', type=str, default='vindr_obj_det_5', help='model name')
    parser.add_argument('--model_name', type=str, default='best', help='model name')
    parser.add_argument('--model_folder', type=str, default='/data/mariammaa/vindr/saved_models/', help='model folder')
    args = parser.parse_args()
    test(args)


main()