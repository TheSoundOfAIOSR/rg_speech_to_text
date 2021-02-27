#!/usr/bin/env python3
"""Capture an audio input into a file. 
In this example there is a time countdown of 60 seconds, 
but as an example for GUI, there is not necessary any limitation..

The soundfile module (https://PySoundFile.readthedocs.io/) must be
installed for this to work.
"""
import sys
sys.path.insert(0, '../..')

import argparse
import asyncio

import sounddevice as sd

from TheSoundOfAIOSR.audiointerface.async_io import capture_to_file


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


async def main(filename, samplerate, channels, subtype, device, buffersize):
    print('capture...')
    audio_task = asyncio.create_task(capture_to_file(
            filename, samplerate, channels, subtype, device, buffersize))
    for i in range(60, 0, -1):
        print(i)
        await asyncio.sleep(1)
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
        'filename', nargs='?', metavar='FILENAME',
        help='audio file to store recording to')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-r', '--samplerate', type=int, help='sampling rate')
    parser.add_argument(
        '-c', '--channels', type=int, default=1, help='number of input channels')
    parser.add_argument(
        '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
    args = parser.parse_args(remaining)

    try:
        if args.samplerate is None:
            device_info = sd.query_devices(args.device, 'input')
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info['default_samplerate'])
        if args.filename is None:
            args.filename = tempfile.mktemp(prefix='delme_capture_to_file_',
                                            suffix='.wav', dir='')

        asyncio.run(main(args.filename,
                         args.samplerate,
                         args.channels,
                         args.subtype,
                         args.device,
                         buffersize=256))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
