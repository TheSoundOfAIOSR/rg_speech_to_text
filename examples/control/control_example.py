#!/usr/bin/env python3
"""Showcase how we integrate STT, TTS, SG, Sampler using 
an async state machine.
"""
import sys
sys.path.insert(0, '.')

import asyncio
import logging
import statesman as sm

from TheSoundOfAIOSR.stt.control.stt_sm import SpeechToTextSM

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(threadName)s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

async def wait_and_show_states(n, t=0.2):
    """ wait_and_show_states is waiting n times a t period, while 
        logs the current state. Useful only to showcase that the main
        thread is not blocked and we can check the current state.
    """
    for _ in range(n):
        logger.debug(await asyncio.sleep(t, result=f"..current state is {control.state}"))

async def silent(coro):
    try:
        await coro
    except RuntimeError as re:
        logger.warning(re)


async def _example_sequential_controller(control):
    """
    Similar hard coded examples can be written for development purpose,
    a chosen event sequence scenario will be executed as it would be
    called by a user.
    """
    await silent(control.load_stt_models())
    await wait_and_show_states(15)

    await silent(control.start_capture_and_transcribe(
        source_device="Mic 1"))
    await wait_and_show_states(5)

    await silent(control.stop_transcription())
    await wait_and_show_states(5)


# we need to have a top level coordinator, a Production state machine
control = SpeechToTextSM()

# then call the _example_sequential_controller, or in case of
# a GUI, or MIDI controller, just hold the top level coordinator
# and when the user events comes in, just call the event methods with await
# as you see in the _example_sequential_controller. 
asyncio.run(_example_sequential_controller(control))
