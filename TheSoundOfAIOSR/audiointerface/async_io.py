import asyncio
import queue

import numpy as np
import sounddevice as sd
import soundfile as sf

async def capture_and_play(buffersize, *, channels=1, dtype='float32',
                           pre_fill_blocks=10, **kwargs):
    """Generator that yields blocks of input/output data as NumPy arrays.

    The output blocks are uninitialized and have to be filled with
    appropriate audio signals.

    """
    assert buffersize != 0
    in_queue = asyncio.Queue()
    out_queue = queue.Queue()
    loop = asyncio.get_running_loop()

    """
    callback(indata: ndarray, outdata: ndarray, frames: int,
         time: CData, status: CallbackFlags) -> None
    """
    def callback(indata, outdata, frame_count, time_info, status):
        loop.call_soon_threadsafe(in_queue.put_nowait, (indata.copy(), status))
        outdata[:] = out_queue.get_nowait()

    # pre-fill output queue
    for _ in range(pre_fill_blocks):
        out_queue.put(np.zeros((buffersize, channels), dtype=dtype))

    stream = sd.Stream(blocksize=buffersize, callback=callback, dtype=dtype,
                       channels=channels, **kwargs)
    with stream:
        while True:
            indata, status = await in_queue.get()
            outdata = np.empty((buffersize, channels), dtype=dtype)
            yield indata, outdata, status
            out_queue.put_nowait(outdata)


async def capture_and_playback(**kwargs):
    """Create a connection between audio inputs and outputs.

    Asynchronously iterates over a stream generator and for each block
    simply copies the input data into the output block.

    """
    async for indata, outdata, status in capture_and_play(**kwargs):
        if status:
            print(status)
        outdata[:] = indata


async def stream_capture(samplerate, channels, device, buffersize, dtype='float32'):
    """Generator that yields blocks of input data 
    captured from sounddevice InputStream into NumPy arrays.

    The audio callback pushes the captured data to `in_queue`,
    and at the same time is consumed the queue and yields the captured data

    """
    assert buffersize != 0
    in_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    """
    callback(indata: numpy.ndarray, frames: int,
         time: CData, status: CallbackFlags) -> None
    """
    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(in_queue.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(samplerate=samplerate,
                            device=device,
                            channels=channels,
                            callback=callback,
                            dtype=dtype,
                            blocksize=buffersize)
    with stream:
        while True:
            indata, status = await in_queue.get()
            yield indata, status


async def capture_to_file(filename, samplerate, channels, subtype,
                          device, buffersize):
    """
    Asynchronously iterates over stream_capture generator and for each block
    simply copies the input data into the audio file.

    """
    # Make sure the file is opened before recording anything:
    with sf.SoundFile(filename, mode='x', samplerate=samplerate,
                      channels=channels, subtype=subtype) as file:

        async for indata, status in stream_capture(samplerate, channels,
                                                   device, buffersize):
            if status:
                print(status)
            file.write(indata)
