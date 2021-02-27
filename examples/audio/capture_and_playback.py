#!/usr/bin/env python3
"""Capture an audio input and immediately play back, using a fixed buffer size.
"""
import sys
sys.path.insert(0, '../..')
     
import asyncio

from TheSoundOfAIOSR.audiointerface.async_io import capture_and_playback

async def main(**kwargs):
    print('capture and play back... (Wait until count down or caancel by keyboard)')
    # create an async audio task
    audio_task = asyncio.create_task(capture_and_playback(**kwargs))

    # we simulate the asynchronous GUI, 
    # waiting 20 seconds or interrupt by keyboard in this example
    for i in range(20, 0, -1):
        print(i)
        await asyncio.sleep(1)

    # Somewhere the GUI cancels the audio task
    audio_task.cancel()
    try:
        await audio_task
    except asyncio.CancelledError:
        print('\nWas cancelled')


if __name__ == "__main__":
    try:
        asyncio.run(main(buffersize=256))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')
