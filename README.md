# Object Detection on Drinks Dataset using RetinaNet with Resnet50 Backbone
This is a project made by Cedric Encarnacion for compliance in EE197Z at UP-EEEI. RetinaNet with Resnet 50 backbone is trained to be able to perform object detection to the drinks dataset. 

## Requirements
Before testing the model, install required packages.
```
pip install -r requirements.txt
```
## Testing
To test the model, simply run
```
python3 test.py
```
By default, this will evaluate the model `./checkpoints/retinanet_resnet50_fpn_drinks_epochs_10.pth`

### **Options**
To test a different model of the same network architecture to a different test dataset.
```
python3 test.py --model <str> --testpath <str>
```
where `--model` the path of the model to be tested and `--testpath` the path of the test dataset to be used.

### **Results**
Below are the performance metric of the model on GPU.
```
Test:  [ 0/51]  eta: 0:01:03  model_time: 1.1525 (1.1525)  evaluator_time: 0.0019 (0.0019)  time: 1.2367  data: 0.0808  max mem: 376
Test:  [50/51]  eta: 0:00:00  model_time: 0.8188 (0.8095)  evaluator_time: 0.0012 (0.0012)  time: 0.8248  data: 0.0001  max mem: 376
Test: Total time: 0:00:41 (0.8148 s / it)
Averaged stats: model_time: 0.8188 (0.8095)  evaluator_time: 0.0012 (0.0012)
Accumulating evaluation results...
DONE (t=0.03s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.877
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.984
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.984
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.809
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.879
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.893
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.908
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.908
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.825
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.911
 ```

## Training
To train, run the command
```
python3 train.py
```
which by default, trains the model at one epoch with batch size 4 and save the weights on the `./checkpoints` directory.

### Options
The script also have the following options
* `--epochs` sets the number of training epochs
* `--batch-size` sets the batch-size for training
* `--datapath` the path of the training/test data
* `--output-path` sets output directory of the model
* `--dont-save-checkpoint` disables saving model to the output path

## Inference
To perform inference, run
```
python3 infer.py --img <img-path> --model <model-path> --testpath <test-path>
```

where `--img` the image to be used as input. By default, inference will be performed randomly on the test dataset.

## Demo
For a video demonstration, run
```
python3 camera_vid.py --index <camera-index>
```
to collect video from an existing camera. Then run
```
python3 demo.py
```
to perform frame-by-frame inference. The output video path is `./demo/demo.mp4`.

## Links
* <a href="https://drive.google.com/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA">Dataset</a>
* Pretrained model: <a href="https://drive.google.com/uc?id=1fiAdIeZZ3at8csSOEYqNWYiKAcn22RS5">retinanet_resnet50_fpn_drinks_epochs_10.pth</a>