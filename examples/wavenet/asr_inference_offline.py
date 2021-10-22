import sys
sys.path.insert(0, '../..')

from TheSoundOfAIOSR.stt.wavenet.inference import WaveNet
from TheSoundOfAIOSR.audiointerface.capture import AudioReader
import argparse
import asyncio

parser = argparse.ArgumentParser(description="ASR with recorded audio (offline)")
parser.add_argument("--recording", "-rec", required=True, help="path to recording file")
parser.add_argument("--model", "-m", default=None, required=False,
                    help="Trained Model path")
parser.add_argument("--tokenizer", "-t", default=None, required=False,
                    help="Trained tokenizer path")
parser.add_argument("--lm", "-l", default=None, required=False,
                    help="Trained lm folder path with unigram and bigram files")
parser.add_argument("--device", "-d", default='cpu', nargs='?', choices=['cuda', 'cpu'], required=False,
                    help="device to use for inferencing")
parser.add_argument("--beam_width", "-bw", default=1, type=int, required=False,
                    help="beam width to use for beam search decoder during inferencing")
args = parser.parse_args()

wavenet = WaveNet(device=args.device, tokenizer_path=args.tokenizer, model_path=args.model,
                  beam_width=args.beam_width, lm_path=args.lm)

print("Loading Models ...")
wavenet.load_model()
print("Models Loaded ...")


async def main():
    loop = asyncio.get_running_loop()
    reader = AudioReader(audio_path=args.recording,
                         sr=16000,
                         dtype="float32")
    inputs, sr = reader.read()
    transcriptions = await wavenet.transcribe(inputs, loop=loop)
    print(transcriptions)


if __name__ == "__main__":
    print("Start Transcribing...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited")
