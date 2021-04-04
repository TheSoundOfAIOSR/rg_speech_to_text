import sys
sys.path.insert(0, '../..')

from TheSoundOfAIOSR.audiointerface import AudioStreaming
from TheSoundOfAIOSR.stt.wavenet import WaveNet
import argparse
import asyncio
import functools

parser = argparse.ArgumentParser(description="ASR with recorded audio")
parser.add_argument("--recording", "-rec", required=True,
                    help="Trained Model path")
parser.add_argument("--model", "-m", default=None,required=False,
                    help="Trained Model path")
parser.add_argument("--tokenizer", "-t", default=None, required=False,
                    help="Trained tokenizer path")
parser.add_argument("--blocksize", "-bs", default=16000, type=int, required=False,
                    help="Size of each audio block to be passed to model")
parser.add_argument("--overlap", "-ov", default=0, type=int, required=False,
                    help="Overlap between blocks")
parser.add_argument("--output", "-out", required=False,
                    help="Output Path for saving resultant transcriptions")
parser.add_argument("--device", "-d", default='cpu', nargs='?', choices=['cuda', 'cpu'], required=False,
                    help="device to use for inferencing")

args = parser.parse_args()

wavenet = WaveNet(device=args.device, tokenizer_path=args.tokenizer, model_path=args.model)

print("Loading Models ...")
wavenet.load_model()
print("Models Loaded ...")

def print_transcription(transcription):
    print(transcription, end=" ")

async def main():
    loop = asyncio.get_running_loop()
    stream = AudioStreaming(audio_path=args.recording, 
                            blocksize=args.blocksize, 
                            overlap=args.overlap, 
                            padding=0, 
                            sr=16000, 
                            dtype="float32")
    async for transcription in wavenet.capture_and_transcribe(stream, loop=loop):
        if not transcription == "":
            print_func = functools.partial(print_transcription, transcription=transcription)
            await loop.run_in_executor(None, print_func)

if __name__=="__main__":
    print("Start Transcribing...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited")