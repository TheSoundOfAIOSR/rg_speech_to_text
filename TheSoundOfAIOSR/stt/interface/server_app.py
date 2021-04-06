#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

import asyncio
import argparse
import logging

from TheSoundOfAIOSR.stt.interface.wsserver import SimpleServerInterface
from TheSoundOfAIOSR.stt.wavenet.inference import WaveNet
from TheSoundOfAIOSR.stt.control.speech_to_text import SpeechToText

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)-8s %(threadName)s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

def stt_main(device, model, tokenizer, frame_len, offline_mode):
    logger.debug("offline_mode=%s", offline_mode)
    stt = SpeechToText(
            WaveNet(device=device, tokenizer_path=tokenizer, model_path=model),
            sample_rate=16000,
            frame_len=frame_len,
            frame_overlap=1, 
            decoder_offset=0,
            channels=1,
            offline_mode=offline_mode)
    srv = SimpleServerInterface(stt=stt, host="localhost", port=8786)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(srv.run())
    except RuntimeError:
        logger.info("Successfully shutdown the Speech To Text service.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR with live audio")
    parser.add_argument("--model", "-m", default=None,required=False,
                        help="Trained Model path")
    parser.add_argument("--tokenizer", "-t", default=None, required=False,
                        help="Trained tokenizer path")
    parser.add_argument("--frame_len", "-fl", default=1.0, type=float, required=False,
                        help="Duration of a buffer in seconds (blocksize = frame_len * sample_rate)")
    parser.add_argument("--device", "-d", default='cpu', nargs='?', choices=['cuda', 'cpu'], required=False,
                        help="device to use for inferencing")
    parser.add_argument("--hostname", "-host", default='localhost', type=str, required=False,
                        help="Host where the websocket server is to be bound")
    parser.add_argument("--port", "-p", default=8786, type=int, required=False,
                        help="Port where the websocket server is to be bound")
    parser.add_argument("--offline_mode", "-ofl", default=1, type=int, required=False,
                        help="Select offline transcription mode.")
    args = parser.parse_args()

    stt_main(args.device, args.model, args.tokenizer, args.frame_len, args.offline_mode == 1)