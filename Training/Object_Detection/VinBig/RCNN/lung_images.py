import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

## Create a dictionary object for each image with its bounding box coords and labels
class LungImages(torch.utils.data.Dataset):
    def __init__(self, pathDir, wp_pathDir, height, width, classes, transforms = None):
       
        self.transforms = transforms
        self.classes = classes
        self.list_filenames = []
        self.list_wp_filenames = []
        self.list_fnames = []
        self.height = height
        self.width = width

        for filename in os.listdir(wp_pathDir):
            if ".png" in filename:  
                self.list_wp_filenames.append(os.path.join(wp_pathDir, filename))             
                orig_filename = filename.replace("png_test_","")
                self.list_filenames.append(os.path.join(pathDir, orig_filename))
                self.list_fnames.append(orig_filename)
    
    ## Overwrite and return the image dictionary
    def __getitem__(self, index):
        imagePath = self.list_filenames[index]
        # Read and resize the image
        origImage = cv2.imread(imagePath)
        image = cv2.cvtColor(origImage, cv2.COLOR_BGR2GRAY).astype(np.float32)
        image = cv2.resize(image, (self.width, self.height), cv2.INTER_AREA)
        image = image / 255.0
      
        wp_imagePath = self.list_wp_filenames[index]
        # Read and resize the image
        wpImage = cv2.imread(wp_imagePath)
        wpImage = cv2.cvtColor(wpImage, cv2.COLOR_BGR2GRAY).astype(np.float32)
        wpImage = cv2.resize(wpImage, (self.width, self.height), cv2.INTER_AREA)
        wpImage = wpImage / 255.0
      
        fname = self.list_fnames[index]
        # Apply data transformations to reduce overfitting
        if self.transforms:
            image = self.transforms(image)
            wpImage = self.transforms(wpImage)
        return fname, image, wpImage
    
    ## Overwrite function to return length of dataset
    def __len__(self):
        return len(self.list_filenames)
