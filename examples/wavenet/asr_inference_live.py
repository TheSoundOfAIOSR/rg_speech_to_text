import sys
sys.path.insert(0, '../..')

from TheSoundOfAIOSR.audiointerface.capture import MicrophoneStreaming
from TheSoundOfAIOSR.stt.wavenet.inference import WaveNet
import argparse
import asyncio
import functools

parser = argparse.ArgumentParser(description="ASR with live audio")
parser.add_argument("--model", "-m", default=None,required=False,
                    help="Trained Model path")
parser.add_argument("--tokenizer", "-t", default=None, required=False,
                    help="Trained tokenizer path")
parser.add_argument("--blocksize", "-bs", default=16000, type=int, required=False,
                    help="Size of each audio block to be passed to model")
parser.add_argument("--output", "-out", required=False,
                    help="Output Path for saving resultant transcriptions")
parser.add_argument("--device", "-d", default='cpu', nargs='?', choices=['cuda', 'cpu'], required=False,
                    help="device to use for inferencing")
parser.add_argument("--pretrained_model_name", "-pwmn", default="facebook/wav2vec2-base-960h", 
                    type=str, required=False, help="Pretrained wavenet model name")

args = parser.parse_args()

wavenet = WaveNet(device=args.device, tokenizer_path=args.tokenizer, model_path=args.model,
                  use_vad=True, pretrained_model_name=args.pretrained_model_name)

print("Loading Models ...")
wavenet.load_model()
print("Models Loaded ...")

def print_transcription(transcription):
    print(transcription, end=" ")
    sys.stdout.flush()

async def main():
    loop = asyncio.get_running_loop()
    stream = MicrophoneStreaming(buffersize=args.blocksize, loop=loop)
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

