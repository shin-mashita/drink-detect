import os
import torch
import csv
import cv2
import tarfile
import gdown
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

# Helper functions

def fetch_pth():
    """Fetch pretrained weight retinanet_resnet50_fpn_drinks_epochs_10.pth"""
    output = './checkpoints/retinanet_resnet50_fpn_drinks_epochs_10.pth'
    if not os.path.exists(output):
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        print("Downloading pretrained weights...")
        url = 'https://drive.google.com/uc?id=1fiAdIeZZ3at8csSOEYqNWYiKAcn22RS5'
        gdown.download(url, output, quiet=False)

def fetch_drinks_dataset():
    """Fetch drinks dataset"""
    if not os.path.exists("./drinks/labels_test.csv") and not os.path.exists("./drinks/labels_train.csv"):
        print("Downloading drinks dataset...")
        url = 'https://drive.google.com/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA'
        output = 'drinks.tar.gz'
        gdown.download(url, output, quiet=False)

        tar = tarfile.open(output,"r:gz")
        tar.extractall()
        tar.close()

        os.remove(output)

def view_prediction(img, prediction, show=False):
    """View predictions as img"""
    img = np.asarray(img)
    img2 = img.copy()
    cmap = [(0,0,0),(0,0,255),(255,0,0),(0,255,0)]
    print(prediction)

    pr = prediction[0]
    boxes = pr['boxes'].cpu().detach().numpy()
    labels = pr['labels'].cpu().detach().numpy()
    scores = pr['scores'].cpu().detach().numpy()
    for box,label,score in zip(boxes,labels,scores):
        if score >= 0.1 and len(box) and label in {0,1,2,3}:
            xmin,ymin,xmax,ymax = [int(b) for b in box]
            cv2.rectangle(img2,(xmin,ymin),(xmax,ymax),cmap[label],2)
    
    if show:
        plt.imshow(img2)
        plt.show()

    return img2

def csv2labels(path):
    """ Build label dict with key = path, value = [xmin, xmax, ymin, ymax, class]"""
    data = []
    with open(path) as f:
        rows = csv.reader(f,delimiter=',')
        for row in rows:
            data.append(row)
    
    data = data[1:]
    data = np.array(data)
    
    mDict = dict()
    for d in data:
        mDict[d[0]]=d[1:]
    
    classes = np.unique(data[:,-1]).astype(int).tolist()
    classes.insert(0,0)

    return mDict,classes

# Initialize Dataset Class
class DrinksDataset(torch.utils.data.Dataset):
    def __init__(self, dictionary, transform=None):
        self.dictionary = dictionary
        self.transform = transform
    
    def __len__(self):
        return len(self.dictionary)
    
    def __getitem__(self,index):
        key = list(self.dictionary.keys())[index]
        xmin,xmax,ymin,ymax,class_id = self.dictionary[key]
        xmin,xmax,ymin,ymax,class_id = int(xmin),int(xmax),int(ymin),int(ymax),int(class_id)

        datapath = "./drinks/"
        img = Image.open(datapath+key).convert("RGB")

        boxes = []
        boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([class_id], dtype=torch.int64)
        image_id = torch.as_tensor([index])
        area = torch.as_tensor([(xmax-xmin)*(ymax-ymin)])
        iscrowd = torch.zeros(1, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img = self.transform(img)
        
        return img,target