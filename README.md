# Transformer-based Keyword Spotting System

A 178K params vision transformer was trained to perform keyword spotting task using the Speech Commands Dataset. The model has a best accuracy of 91.10% on the test split. 

This is an assignment made by Cedric Encarnacion for compliance in 197Z. 

## Requirements
Before training/testing, install required dependencies.
```
pip install -r requirements.txt
```
## Training
To train the model, run
```
python3 train.py
```

### Options
The script allows some parameters of the model to be modified. Options available are `--depth`, `--embed-dim`, `--num-heads`, `--patch-num`, and `--num-classes`.

Before the model processes it, the script converts the audio file into mel spectrogram which parameters can be modified using `--n-fft`, `--n-mels`, `--win-length`, and `--hop-length`.

Training parameters can also be changed.
* `--max-epochs` sets the number of training epochs
* `--batch-size` sets the batch-size for training
* `--data-path` the path of the training/test data
* `--lr` the learning rate for training
* `--accelerator` device used for training
* `--num-workers` the number of workers for each of the dataloaders

## Testing
To get the performance of the trained model, run
```
python3 test.py
```

### **Options**
Training and dataset parameters can be modified using the options for training. 

### **Results**
Below shows the test accuracy and test loss of the model on GPU.
```
─────────────────────────────────────────────────
       Test metric             DataLoader 0
─────────────────────────────────────────────────
        test_acc             91.10244750976562
        test_loss           0.38203972578048706

 ```


## Inference
To perform inference, run
```
python3 kws-infer.py
```
By default, a keyword-spotting GUI will pop up and the user is required to input keywords through a microphone. 


## Links
* <a href="https://github.com/shin-mashita/drink-detect/releases/download/v1.0.1/kws_transformer_91_1129.pt">Trained model</a>
