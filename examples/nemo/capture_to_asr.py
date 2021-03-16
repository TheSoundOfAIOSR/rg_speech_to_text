#!/usr/bin/env python3
"""Capture an audio input into ASR. 
In this example there is a time countdown of 60 seconds, 
but as an example for GUI, there is not necessary any limitation..
"""
import sys
sys.path.insert(0, '../..')

import argparse
import asyncio

import sounddevice as sd

from TheSoundOfAIOSR.stt.nemo.streaming_capture import capture_and_transcribe
from TheSoundOfAIOSR.stt.nemo.streaming_model import create_quartznet_streaming_asr


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


async def main(device):
    print('create QuartzNet streaming ASR...')
    # sample rate, Hz
    SAMPLE_RATE = 16000
    # number of audio channels (expect mono signal)
    CHANNELS = 1
    # duration of signal frame, seconds
    FRAME_LEN = 1.0
    # frame overlap
    FRAME_OVERLAP = 2
    # decoder offset
    OFFSET = 4
    asr = create_quartznet_streaming_asr(sample_rate=SAMPLE_RATE,
                                     frame_len=FRAME_LEN,
                                     frame_overlap=FRAME_OVERLAP,
                                     offset=OFFSET)
    audio_task = asyncio.create_task(capture_and_transcribe(
            asr, SAMPLE_RATE, CHANNELS, device,
            int(FRAME_LEN * SAMPLE_RATE), offset=OFFSET))
    print('capture...')
    for i in range(600, 0, -1):
        sys.stdout.flush()
        await asyncio.sleep(0.1)
    audio_task.cancel()
    try:
        await audio_task
    except asyncio.CancelledError:
        print('\nwire was cancelled')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    args = parser.parse_args(remaining)

    try:
        asyncio.run(main(args.device))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
