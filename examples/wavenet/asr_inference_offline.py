import sys
sys.path.insert(0, '../..')

from TheSoundOfAIOSR.stt.wavenet.inference import WaveNet
import argparse
import soundfile as sf
from scipy.signal import resample
import torch
from torchaudio.transforms import Resample

parser = argparse.ArgumentParser(description="ASR with recorded audio (offline)")
parser.add_argument("--recording", "-rec", required=True)
args = parser.parse_args()

wavenet = WaveNet(device='cpu')

print("Loading Models ...")
wavenet.load_model()
print("Models Loaded ...")

USE_TORCHAUDIO_RESAMPLING = True

def main():
    wav, sr = sf.read(args.recording)
    target_sr = 16000
    if USE_TORCHAUDIO_RESAMPLING:
        resampling_transform = Resample(orig_freq=sr,
                                    new_freq=target_sr)

        inputs = resampling_transform(torch.Tensor([wav])).squeeze()
    else:
        inputs = resample(wav, num=int(len(wav)*target_sr/sr))

    print(wavenet.transcribe(inputs))

if __name__=="__main__":
    print("Start Transcribing...")
    main()