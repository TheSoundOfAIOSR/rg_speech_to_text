import asyncio

from TheSoundOfAIOSR.audiointerface.async_io import stream_capture

async def capture_and_transcribe(
        asr, samplerate, channels, device, buffersize, offset):

    empty_counter = 0

    async for indata, status in stream_capture(samplerate, channels,
                                               device, buffersize, dtype='int16'):
        if status:
            print(status)
        # indata has shape for multiple channels, for example (n,1) for mono,
        # but we need a shape for strictly one channel, like (n,) 
        signal = indata.reshape(indata.shape[0])
        text = asr.transcribe(signal)
        if len(text):
            print(text,end='')
            empty_counter = offset
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                print(' ',end='')