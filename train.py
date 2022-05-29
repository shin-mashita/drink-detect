import os

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from kws_transformer import LitTransformer
from datamodule import KWSDataModule


def get_args():
    parser = ArgumentParser(description='KWS Transformer')

    # Transformer params
    parser.add_argument('--depth', type=int, default=12, help='depth')
    parser.add_argument('--embed-dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='num_heads')
    parser.add_argument('--patch-num', type=int, default=16, help='patch_num')
    parser.add_argument("--num-classes", type=int, default=37)

    # Mel spectrogram params
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)

    # Training params
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: )')
    parser.add_argument('--max-epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0)')
    parser.add_argument('--data-path', default="./data/speech_commands/", type=str, metavar='N')
    parser.add_argument('--accelerator', default='gpu', type=str, metavar='N')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N')
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args

def debug(show=True,model=None):
    if args.debug == True:
        print()
        print("----- Model params -----")
        print("Embed dim: ", args.embed_dim)
        print("Patch nums: ", args.patch_num)
        print("Depth: ", args.depth)
        print("Heads: ", args.num_heads)
        print("Seq len: ", seqlen)
        print()
        print("----- Testing params -----")
        print("Device: ", args.accelerator)
        print("Epochs: ", args.max_epochs)
        print("Batch size: ", args.batch_size)
        print()

if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

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
                                patch_num=args.patch_num,
                                class_dict=CLASS_TO_IDX)
    datamodule.setup()

    data = iter(datamodule.train_dataloader()).next()
    patch_dim = data[0].shape[-1]
    seqlen = data[0].shape[-2]

    model = LitTransformer( num_classes=len(CLASSES), 
                            lr=args.lr,
                            epochs=args.max_epochs, 
                            depth=args.depth,
                            embed_dim=args.embed_dim,
                            head=args.num_heads,
                            patch_dim=patch_dim,
                            seqlen=seqlen)

    trainer = Trainer(  accelerator=args.accelerator, max_epochs=args.max_epochs, 
                        precision=16 if args.accelerator == 'gpu' else 32)
    
    debug(args.debug)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)