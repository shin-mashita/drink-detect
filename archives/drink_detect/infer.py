import os
import torch
import torchvision
import argparse
import numpy as np

from PIL import Image
from torchvision import transforms as T
from drinks_utils import csv2labels, view_prediction, fetch_drinks_dataset, fetch_pth

def get_args():
    parser = argparse.ArgumentParser(description="Drinks object detection")
    parser.add_argument("--img", default="", type=str)
    parser.add_argument("--model", default="./checkpoints/retinanet_resnet50_fpn_drinks_epochs_10.pth", type=str)
    parser.add_argument("--testpath", default="./drinks", type=str)
    return parser

def main(args):
    # Fetch dataset and pretrained model
    fetch_drinks_dataset()
    fetch_pth()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        "img":args.img,
        "model":args.model,
        "testpath":args.testpath
    }


    if not os.path.exists(config["img"]) or config["img"] == "":
        labels,_ = csv2labels(os.path.join(config["testpath"],"labels_test.csv"))
        labels = list(labels.keys())
        imgpath = os.path.join(config["testpath"],labels[np.random.randint(0,51)])
    
    img = Image.open(imgpath).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    input = transform(img)

    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.load_state_dict(torch.load(config["model"]))
    model.to(device)
    model.eval()

    with torch.no_grad():
        prediction = model([input.to(device)])
    print(prediction)

    view_prediction(img,prediction,show=True)

if __name__ == "__main__":
    args = get_args().parse_args()
    main(args)