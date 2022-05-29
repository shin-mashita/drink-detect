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
    cmap = [(0,0,0),(255,0,0),(0,0,255),(0,255,0)]
    label_maps ={
        1: "Summit Drinking Water 500ml",
        2: "Coca-Cola 330ml",
        3: "Del Monte 100% Pineapple Juice 240ml"
    }

    print(prediction)

    pr = prediction[0]
    boxes = pr['boxes'].cpu().detach().numpy()
    labels = pr['labels'].cpu().detach().numpy()
    scores = pr['scores'].cpu().detach().numpy()

    newboxes = []
    newlabels = []
    newscores = []

    if len(boxes)>1:
        c = torch.corrcoef(pr['boxes'])

        t1 = (c >= 0.98).nonzero(as_tuple=True)[0].cpu().numpy()
        t2 = (c >= 0.98).nonzero(as_tuple=True)[1].cpu().numpy()

        to_remove = set()
        for i1,i2 in zip(t1,t2):
            if i1 == i2:
                continue
            else:
                if scores[i1]>=scores[i2]:
                    to_remove.add(i2)
                else:
                    to_remove.add(i1)

        for i in range(len(boxes)):
            if i not in to_remove:
                newboxes.append(boxes[i])
                newlabels.append(labels[i])
                newscores.append(scores[i])
    else:
        newboxes = boxes.copy()
        newlabels = labels.copy()
        newscores = scores.copy()

    for box,label,score in zip(newboxes,newlabels,newscores):
        if len(box) and label in {0,1,2,3}:
            xmin,ymin,xmax,ymax = [int(b) for b in box]
            cv2.rectangle(img2,(xmin,ymin),(xmax,ymax),cmap[label],2)
            cv2.putText(img2,label_maps[label],(xmin,ymax+10),0,0.3,cmap[label])

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