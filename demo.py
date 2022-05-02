import torchvision
import os
import torch
import cv2
import numpy as np

from torchvision import transforms as T

def view_prediction(img, prediction):
    img = np.asarray(img)
    img2 = img.copy()
    cmap = [(0,0,0),(0,0,255),(255,0,0),(0,255,0)]

    pr = prediction[0]
    boxes = pr['boxes'].cpu().detach().numpy()
    labels = pr['labels'].cpu().detach().numpy()
    for box,label in zip(boxes,labels):
        if len(box) and label in {0,1,2,3}:
            xmin,ymin,xmax,ymax = [int(b) for b in box]
            cv2.rectangle(img2,(xmin,ymin),(xmax,ymax),cmap[label],2)
    
    return img2


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model_dir = "./checkpoints"
    model_name = "retinanet_resnet50_fpn_drinks_epochs_10.pth" # NOTE: epochs_3.pth perform better for summit
    model_path = os.path.join(model_dir,model_name)
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        # fetch_pth()

    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    transform = T.Compose([T.ToTensor()])

    in_path = "./demo/input.avi"
    vidpath = "./demo/demo.avi"
    if not os.path.exists("./demo"):
        os.mkdir("./demo")

    cap = cv2.VideoCapture(in_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

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
    print("Inference done. Check ./demo/demo.avi")


if __name__ == "__main__":
    main()