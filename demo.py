import torchvision
import os
import torch
import cv2
import numpy as np

from torchvision import transforms as T
from drinks_utils import fetch_pth, fetch_drinks_dataset, view_prediction

def main():
    # Fetch dataset and pretrained model
    fetch_drinks_dataset()
    fetch_pth()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model_dir = "./checkpoints"
    model_name = "retinanet_resnet50_fpn_drinks_epochs_10.pth"
    model_path = os.path.join(model_dir,model_name)

    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    transform = T.Compose([T.ToTensor()])

    in_path = "./demo/input.mp4"
    vidpath = "./demo/demo.mp4"
    if not os.path.exists("./demo"):
        os.mkdir("./demo")

    cap = cv2.VideoCapture(in_path)
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")

    out = cv2.VideoWriter(vidpath,fourcc, 10.0, (640,480))
    
    while(True):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        input = transform(frame)
        with torch.no_grad():
            prediction = model([input.to(device)])
        img = view_prediction(frame,prediction)
        out.write(img)

    cap.release()
    out.release()
    print("Inference done. Check ./demo/demo.mp4")


if __name__ == "__main__":
    main()