import os
import torch
import torchvision
import csv
import utils
import argparse
import numpy as np

from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as T
from engine import evaluate
from drinks_utils import DrinksDataset, csv2labels, fetch_drinks_dataset, fetch_pth

# Helper Functions
def get_args():
    parser = argparse.ArgumentParser(description="Drinks object detection")
    parser.add_argument("--model", default="./checkpoints/retinanet_resnet50_fpn_drinks_epochs_10.pth", type=str)
    parser.add_argument("--testpath", default="./drinks", type=str)    
    return parser

def main(args):
    # Fetch dataset and pretrained weights
    fetch_drinks_dataset()
    fetch_pth()

    config = {
        "model":args.model,
        "testpath":args.testpath
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_csv_path = os.path.join(config["testpath"],"labels_test.csv")
    test_dict, test_classes = csv2labels(test_csv_path)

    # Initialize Data Loader for training/ testing
    transform = T.Compose([T.ToTensor()])
    test_split = DrinksDataset(test_dict, transform)
    test_loader = DataLoader(   test_split,
                                batch_size=1,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True,
                                collate_fn=utils.collate_fn)

    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.load_state_dict(torch.load(config["model"]))
    model.to(device)

    evaluate(model, test_loader, device=device)


if __name__ == "__main__":
    args = get_args().parse_args()
    main(args)