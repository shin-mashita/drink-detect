import torch
import argparse
import torchaudio
import os
import numpy as np
import librosa
import sounddevice as sd
import time
import validators

from einops import rearrange
from torchvision.transforms import ToTensor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./data/speech_commands/")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--wav-file", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="https://github.com/shin-mashita/drink-detect/releases/download/v1.0.1/kws_transformer_91_1129.pt")
    parser.add_argument("--gui", default=False, action="store_true")
    parser.add_argument("--rpi", default=False, action="store_true")
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    idx_to_class = {i: c for i, c in enumerate(CLASSES)}

    args = get_args()

    if validators.url(args.checkpoint):
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        checkpoint = os.path.join('./checkpoints',args.checkpoint.rsplit('/', 1)[-1])
        if not os.path.isfile(checkpoint):
            print('Downloading',args.checkpoint.rsplit('/', 1)[-1])
            torch.hub.download_url_to_file(args.checkpoint, checkpoint)
    else:
        checkpoint = args.checkpoint


    print("Loading model checkpoint: ", checkpoint)
    model = torch.load(checkpoint)

    if args.gui:
        import PySimpleGUI as sg
        sample_rate = 16000
        sd.default.samplerate = sample_rate
        sd.default.channels = 1
        sg.theme('DarkTanBlue')
  
    elif args.wav_file is None:
        # list wav files given a folder
        print("Searching for random kws wav file...") # TODO: Limit to test dataset
        label = CLASSES[2:]
        label = np.random.choice(label)
        path = os.path.join(args.path, "SpeechCommands/speech_commands_v0.02/")
        path = os.path.join(path, label)
        wav_files = [os.path.join(path, f)
                     for f in os.listdir(path) if f.endswith('.wav')]
        # select random wav file
        wav_file = np.random.choice(wav_files)
        print('Inference on: ', wav_file)

    else:
        wav_file = args.wav_file
        label = args.wav_file.split("/")[-1].split(".")[0]
    
    if not args.gui:
        waveform, sample_rate = torchaudio.load(wav_file)

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                     n_fft=args.n_fft,
                                                     win_length=args.win_length,
                                                     hop_length=args.hop_length,
                                                     n_mels=args.n_mels,
                                                     power=2.0)
    if not args.gui:
        if waveform.shape[-1] < sample_rate:
            waveform = torch.cat([waveform, torch.zeros((1, sample_rate - waveform.shape[-1]))], dim=-1)
        elif waveform.shape[-1] > sample_rate:
            waveform = waveform[:,:sample_rate]

        mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
        mel = rearrange(mel, 'b m (p t) -> b (p) (m t)', p=16)

        pred = torch.argmax(model(mel), dim=1)
        print(f"Ground Truth: {label}, Prediction: {idx_to_class[pred.item()]}")
        exit(0)

    layout = [ 
        [sg.Text('Say it!', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 140), key='-OUTPUT-'),],
        [sg.Text('', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 100), key='-STATUS-'),],
        [sg.Text('Speed', expand_x=True, font=("Helvetica", 28), key='-TIME-')],
        [sg.Text('Confidence', expand_x=True, font=("Helvetica", 28), key='-ACC-')]
    ]

    window = sg.Window('KWS Inference', layout, location=(0,0), resizable=True).Finalize()
    window.Maximize()
    window.BringToFront()

    total_runtime = 0
    n_loops = 0
    while True:
        event, values = window.read(100)
        if event == sg.WIN_CLOSED:
            break
        
        waveform = sd.rec(sample_rate).squeeze()
        
        sd.wait()
        if waveform.max() > 1.0:
            continue
        start_time = time.time()
        if args.rpi:
            # this is a workaround for RPi 4
            # torch 1.11 requires a numpy >= 1.22.3 but librosa 0.9.1 requires == 1.21.5
            waveform = torch.FloatTensor(waveform.tolist())
            mel = np.array(transform(waveform).squeeze().tolist())
            mel = librosa.power_to_db(mel, ref=np.max).tolist()
            
            mel = torch.FloatTensor(mel)
            mel = mel.unsqueeze(0)

        else:
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))

        mel = mel.unsqueeze(0)
        mel = rearrange(mel, 'b c m (p t) -> b (p) (c m t)', p=16)
        pred = model(mel)
        pred = torch.functional.F.softmax(pred, dim=1)
        max_prob =  pred.max()
        elapsed_time = time.time() - start_time
        total_runtime += elapsed_time
        n_loops += 1
        ave_pred_time = total_runtime / n_loops
        window['-TIME-'].update(f"{ave_pred_time:.2f} sec")
        window['-ACC-'].update(f"{max_prob*100:.2f}%")

        if max_prob > args.threshold:
            pred = torch.argmax(pred, dim=1)
            human_label = f"{idx_to_class[pred.item()]}"
            window['-OUTPUT-'].update(human_label)
            window['-OUTPUT-'].update(human_label)
            if human_label == "stop":
                window['-STATUS-'].update("Goodbye!")
                # refresh window
                window.refresh()
                time.sleep(1)
                break
                
        else:
            window['-OUTPUT-'].update("...")
        
        window['-TIME-'].update(f"{ave_pred_time:.2f} sec")
        window['-ACC-'].update(f"{max_prob*100:.2f}%")


    window.close()