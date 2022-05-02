import os
import torch
import torchvision
import utils
import argparse

from torch.utils.data import DataLoader
from torchvision import transforms as T
from engine import train_one_epoch, evaluate
from drinks_utils import DrinksDataset, csv2labels, fetch_drinks_dataset

def get_args():
    parser = argparse.ArgumentParser(description="Drinks object detection")
    parser.add_argument("--epochs", default=3, type=int, metavar="N")
    parser.add_argument("--batch-size", default=4, type=int, metavar="N")
    parser.add_argument("--datapath", default="./drinks", type=str)
    parser.add_argument("--output-path", default="./checkpoints", type=str)
    parser.add_argument("--dont-save-checkpoint", action="store_true")
    return parser

def main(args):
    # Fetch dataset
    fetch_drinks_dataset()

    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "datapath":args.datapath,
        "output_path":args.output_path,
        "dont_save_checkpoint":args.dont_save_checkpoint
    }

    # Generate label dict from csv
    train_csv_path = os.path.join(config["datapath"],"labels_train.csv")
    test_csv_path = os.path.join(config["datapath"],"labels_test.csv")
    
    train_dict, train_classes = csv2labels(train_csv_path)
    test_dict, test_classes = csv2labels(test_csv_path)

    # Initialize Data Loader for training/ testing
    transform = T.Compose([T.ToTensor()])
    train_split = DrinksDataset(train_dict, transform)
    test_split = DrinksDataset(test_dict, transform)

    train_loader = DataLoader(  train_split,
                                batch_size=config["batch_size"],
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True,
                                collate_fn=utils.collate_fn)

    test_loader = DataLoader(   test_split,
                                batch_size=1,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True,
                                collate_fn=utils.collate_fn)
    
    # Initialize model
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.to(device)

    # Initialize optimizer and lr scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop 
    epochs = config["epochs"]

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)
        lr_scheduler.step()
    
    evaluate(model, test_loader, device=device)
    
    # Save model 
    if not os.path.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    if not config["dont_save_checkpoint"]:
        fpath = 'retinanet_resnet50_fpn_drinks_epochs_' + str(args.epochs) + '.pth'
        fpath = os.path.join(config["output_path"],fpath)
        torch.save(model.state_dict(),fpath)
    

if __name__ == "__main__":
    args = get_args().parse_args()
    main(args)
    
