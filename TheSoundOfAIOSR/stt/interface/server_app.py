#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

import asyncio
import argparse
import logging

from TheSoundOfAIOSR.stt.interface.wsserver import SimpleServerInterface
from TheSoundOfAIOSR.stt.control.speech_to_text import SpeechToText

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)-8s %(threadName)s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

def stt_main_wavenet(device, model, tokenizer, frame_len, offline_mode,
                     use_vad, pretrained_wavenet_model_name):
    """
    Args:
        device: 'cpu' or 'cuda'
        mode: model
        tokenizer: tokenizer
        frame_len: duration of signal frame, seconds
        offline_mode: offline mode
        use_vad: use voice activity detection
        pretrained_wavenet_model_name: model name
    """
    from TheSoundOfAIOSR.stt.wavenet.inference import WaveNet

    logger.debug("offline_mode=%s", offline_mode)
    logger.debug("use_vad=%s", use_vad)
    # sample rate, Hz
    SAMPLE_RATE = 16000
    # frame overlap
    FRAME_OVERLAP = 1
    stt = SpeechToText(
            WaveNet(device=device,
                    tokenizer_path=tokenizer,
                    model_path=model,
                    use_vad=use_vad,
                    pretrained_model_name=pretrained_wavenet_model_name),
            block_size=int(frame_len * SAMPLE_RATE * FRAME_OVERLAP),
            offline_mode=offline_mode)
    srv = SimpleServerInterface(stt=stt, host="localhost", port=8786)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(srv.run())
    except RuntimeError:
        logger.info("Successfully shutdown the Speech To Text service.")


def stt_main_nemo(device, frame_len, offline_mode):
    """
    Args:
        device: 'cpu' or 'cuda'
        frame_len: duration of signal frame, seconds
        offline_mode: offline mode
    """
    from TheSoundOfAIOSR.stt.nemo.inference import NemoASR

    logger.debug("offline_mode=%s", offline_mode)
    # sample rate, Hz
    SAMPLE_RATE = 16000
    # frame overlap
    FRAME_OVERLAP = 2
    # decoder offset
    OFFSET = 4
    stt = SpeechToText(
            NemoASR(sample_rate=SAMPLE_RATE,
                    frame_len=frame_len,
                    frame_overlap=FRAME_OVERLAP,
                    offset=OFFSET,
                    device=device),
            block_size=int(frame_len * SAMPLE_RATE * FRAME_OVERLAP),
            offline_mode=offline_mode)
    srv = SimpleServerInterface(stt=stt, host="localhost", port=8786)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(srv.run())
    except RuntimeError:
        logger.info("Successfully shutdown the Speech To Text service.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR with live audio")
    parser.add_argument("--family", "-F", default='wavenet', choices=['wavenet', 'nemo'], required=False,
                        help="Family of model: 'wavenet' or 'nemo'"),
    parser.add_argument("--model", "-m", default=None,required=False,
                        help="Trained Model path (for WaveNet)")
    parser.add_argument("--tokenizer", "-t", default=None, required=False,
                        help="Trained tokenizer path (for WaveNet)")
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
    parser.add_argument("--use_vad", "-vad", default=0, type=int, required=False,
                        help="Use Voice Activity Detection.")
    parser.add_argument("--pretrained_wavenet_model_name", "-pwmn", default="iamtarun/wav2vec-osr", 
                        type=str, required=False, help="Pretrained wavenet model name. eg: facebook/wav2vec2-base-960h")
    args = parser.parse_args()

    if args.family == 'wavenet':
        stt_main_wavenet(args.device, args.model, args.tokenizer, args.frame_len,
            args.offline_mode == 1, args.use_vad == 1, args.pretrained_wavenet_model_name)
    elif args.family == 'nemo':
        stt_main_nemo(args.device, args.frame_len, args.offline_mode == 1)

