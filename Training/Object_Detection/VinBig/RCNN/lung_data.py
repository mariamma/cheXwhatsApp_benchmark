import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

## Create a dictionary object for each image with its bounding box coords and labels
class LungData(torch.utils.data.Dataset):
    def __init__(self, height, width, xBox, yBox, wBox, hBox, 
        boxLabel, pngImages, classes, transforms = None, wpPath = None):
        self.height = height
        self.width = width
        self.len = len(boxLabel)
        self.transforms = transforms

        self.pngImages = pngImages
        self.boxLabel = boxLabel
        self.xBox = xBox
        self.yBox = yBox
        self.wBox = wBox
        self.hBox = hBox
        self.classes = classes

        self.WpPath = wpPath
    
    ## Overwrite and return the image dictionary
    def __getitem__(self, index):
        imagePath = self.pngImages['Directory'].iloc[index]
        
        # Read and resize the image
        origImage = cv2.imread(imagePath)
        image = cv2.cvtColor(origImage, cv2.COLOR_BGR2GRAY).astype(np.float32)
        image = cv2.resize(image, (self.width, self.height), cv2.INTER_AREA)
        image = image / 255.0

        if self.WpPath != None:
            fname = imagePath.split("/")[-1]
            whatsappImage = cv2.imread(os.path.join(self.WpPath, fname))
            whatsappImage = cv2.cvtColor(whatsappImage, cv2.COLOR_BGR2GRAY).astype(np.float32)
            whatsappImage = cv2.resize(whatsappImage, (self.width, self.height), cv2.INTER_AREA)
            whatsappImage = whatsappImage / 255.0
        
        ## Combine all boxes for an image together
        boxes = []
        labels = []   

        Wimage = origImage.shape[1]
        Himage = origImage.shape[0]

        ## Create dictionary with image info and boxes
        for member in range(len(self.boxLabel['Target'].iloc[index])):
            labels.append(self.classes.index(self.boxLabel['Target'].iloc[index][member]))
            
            xMin = self.xBox[index][member]
            xMax = self.xBox[index][member] + self.wBox[index][member]    
            yMin = self.yBox[index][member]
            yMax = self.yBox[index][member] + self.hBox[index][member]
            
            xMinCorr = (xMin/Wimage) * self.width
            xMaxCorr = (xMax/Wimage) * self.width
            yMinCorr = (yMin/Himage) * self.height
            yMaxCorr = (yMax/Himage) * self.height
            
            boxes.append([xMinCorr, yMinCorr, xMaxCorr, yMaxCorr])
        
        ## Convert bounding boxes to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        ## Calculate area of boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        ## Suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        ## Create output dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        imageId = torch.tensor([index])
        # target["image_id"] = imageId
        target["image_id"] = index
        
        target["image_path"] = imagePath
        
        # Apply data transformations to reduce overfitting
        if self.transforms:
            image = self.transforms(image)
            
            # image = sample['image']
            # target['boxes'] = torch.Tensor(sample['bboxes'])
               
        if self.WpPath != None:
            return fname, image, whatsappImage, target        
        return image, target
    
    ## Overwrite function to return length of dataset
    def __len__(self):
        return self.len


# ## Check dataset length
# dataset = LungData(256, 256)
# print("Length of dataset: ", len(dataset), "\n")

# ## Sample index to show image dictionary output
# image, target = dataset[878]
# print("Image shape: ", image.shape, "\n")
# print("Image dictionary object: ", target)