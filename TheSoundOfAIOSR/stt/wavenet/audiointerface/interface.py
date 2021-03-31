import time
import asyncio
import pyaudio
import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample

class MicrophoneStreaming_sounddevice:
    def __init__(self, sr=16000, buffersize=1024, channels=1, device=None, loop=None, dtype="float32"):
        assert buffersize >=0, "buffersize cannot be 0 or negative"
        self._sr = sr
        self._channels = channels
        self._device = device
        self._buffer = asyncio.Queue()
        self._buffersize = buffersize
        self._dtype = dtype
        self._loop = loop
    
    def __callback(self, indata, frame_count, time_info, status):
        self._loop.call_soon_threadsafe(self._buffer.put_nowait, (indata.copy(), status))
    
    async def record_to_file(self, filename, duration=None):
        with sf.SoundFile(filename, mode='x', samplerate=self._sr, channels=self._channels) as f:
            t = time.time()
            rec = duration if duration is not None else 10
            async for block, status in self.generator():
                f.write(block)
                rec = duration+0 if duration is not None else duration+1
                if(time.time() - t) > rec:
                    break 

    async def generator(self):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        stream = sd.InputStream(
            samplerate=self._sr,
            device=self._device,
            channels=self._channels,
            callback=self.__callback,
            dtype=self._dtype,
            blocksize=self._buffersize
            )
        with stream:
            while True:
                try:
                    indata, status = await self._buffer.get()
                    yield indata.squeeze(), status
                except asyncio.QueueEmpty:
                    self._loop.stop()
                    break

class MicrophoneStreaming_pyaudio:
    def __init__(self, sr=16000, buffersize=1024, channels=1, loop=None, dtype="float32"):
        assert buffersize >= 0, "buffersize cannot be 0 or negative"
        self._sr = sr
        self._dtype = dtype
        self._buffersize = buffersize
        self._buffer = asyncio.Queue()
        self._channels = channels
        self._closed = True
        self._loop = loop

    async def __aenter__(self):
        await self.__open()
        return self
    
    async def __aexit__(self, type, value, traceback):
        await self.__close()
    
    def __callback(self, in_data, frame_count, time_info, status):
        self._loop.call_soon_threadsafe(self._buffer.put_nowait, (in_data, status))
        return None, pyaudio.paContinue

    async def __open(self):
        self._pyaudio_obj = pyaudio.PyAudio()
        self._stream = self._pyaudio_obj.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self._sr,
            input=True,
            frames_per_buffer=self._buffersize,
            stream_callback=self.__callback
        )
        self._closed = False

    async def __close(self):
        self._stream.stop_stream()
        self._stream.close()
        self._closed = True
        self._pyaudio_obj.terminate()
        del(self._buffer)

    async def record_to_file(self, filename, duration=None):
        with sf.SoundFile(filename, mode='x', samplerate=self._sr, channels=self._channels) as f:
            t = time.time()
            rec = duration if duration is not None else 10
            async for block, status in self.generator():
                f.write(block)
                rec = duration+0 if duration is not None else duration+1
                if(time.time() - t) > rec:
                    break 

    async def generator(self):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        while not self._closed:
            try:
                chunk, status = await self._buffer.get()
                yield np.fromstring(chunk, dtype=self._dtype), status
            except asyncio.QueueEmpty:
                self._loop.stop()
                break

class MicrophoneStreaming:
    def __init__(self, sr=16000, buffersize=1024, channels=1, device=None, loop=None, interface="sd", dtype="float32"):
        assert self.__check_valid_interface(interface), "interface can be sd (sounddevice) or pyaudio"
        self._interface = interface
        self._sr = sr
        self._dtype = dtype
        self._buffersize = buffersize
        self._channels = channels
        self._device = device
        self._loop = loop

    def stream(self):
        if self._interface=="sd":
            return MicrophoneStreaming_sounddevice(self._sr, self._buffersize, self._channels, self._device, self._loop, self._dtype)
        elif self._interface=="pyaudio":
            return MicrophoneStreaming_pyaudio(self._sr, self._buffersize, self._channels, self._loop, self._dtype)

    def __check_valid_interface(self, interface):
        if interface in ["sd", "pyaudio"]:
            return True
        return False

class AudioStreaming:
    def __init__(self, audio_path, blocksize, sr=16000, overlap=0, padding=None, dtype="float32"):
        assert blocksize >= 0, "blocksize cannot be 0 or negative"
        self._sr = sr
        self._orig_sr = sf.info(audio_path).samplerate
        self._sf_blocks = sf.blocks(audio_path,
                        blocksize=blocksize, 
                        overlap=overlap,
                        fill_value=padding,
                        dtype=dtype)

    async def generator(self):
        for block in self._sf_blocks:
            chunk = await self.__resample_file(block, self._orig_sr, self._sr)
            yield chunk, self._orig_sr

    async def __resample_file(self, array, original_sr, target_sr):
        sample = resample(array, num=int(len(array)*target_sr/original_sr))
        return sample


class AudioReader:
    def __init__(self, audio_path, sr=16000, dtype="float32"):
        self._sr = sr
        self._dtype = dtype
        self._audio_path = audio_path
    
    def read(self):
        data, sr = sf.read(self._audio_path, dtype=self._dtype)
        data = self.__resample_file(data, sr, self._sr)
        return data, sr

    def __resample_file(self, array, original_sr, target_sr):
        return resample(array, num=int(len(array)*target_sr/original_sr))