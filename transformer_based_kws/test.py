import torch
import os
import validators

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from kws_transformer import LitTransformer
from datamodule import KWSDataModule
from torchsummary import summary


def get_args():
    parser = ArgumentParser(description='KWS Transformer')

    # Mel spectrogram params
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)

    # Testing params
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: )')
    parser.add_argument('--data-path', default="./data/speech_commands/", type=str, metavar='N')
    parser.add_argument('--accelerator', default='gpu', type=str, metavar='N')
    parser.add_argument('--checkpoint', default='https://github.com/shin-mashita/drink-detect/releases/download/v1.0.1/kws_transformer_91_1129.pt', type=str, metavar='N')
    parser.add_argument('--num-workers', default=4, type=int, metavar='N')
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


def debug(show=False, model=None):
    if args.debug == True:
        print()
        print("----- Testing params -----")
        print("Device: ", args.accelerator)
        print("Ckpt path/url: ", args.checkpoint)
        print("Batch size: ", args.batch_size)
        print()
        if model is not None:
            device = 'cuda' if args.accelerator=='gpu' else 'cpu'
            summary(model.to(device),(16,256))


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    if validators.url(args.checkpoint):
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        checkpoint = os.path.join('./checkpoints',args.checkpoint.rsplit('/', 1)[-1])
        if not os.path.isfile(checkpoint):
            print('Downloading',args.checkpoint.rsplit('/', 1)[-1])
            torch.hub.download_url_to_file(args.checkpoint, checkpoint)
    else:
        checkpoint = args.checkpoint

    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
            'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
            'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
            'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    datamodule = KWSDataModule( path=args.data_path,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                n_mels=args.n_mels,
                                n_fft=args.n_fft,
                                win_length=args.win_length,
                                hop_length=args.hop_length,
                                patch_num=16,
                                class_dict=CLASS_TO_IDX)
    datamodule.setup()

    model = torch.load(checkpoint)

    trainer = Trainer(accelerator=args.accelerator, max_epochs=-1)

    debug(args.debug,model)
    trainer.test(model, datamodule=datamodule)

    