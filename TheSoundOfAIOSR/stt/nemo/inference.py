import asyncio
import functools

from TheSoundOfAIOSR.stt.nemo.streaming_capture import capture_and_transcribe
from TheSoundOfAIOSR.stt.nemo.streaming_model import create_quartznet_streaming_asr

class NemoASR:
    def __init__(self, sample_rate, frame_len, frame_overlap, offset, device="cpu"):
        '''
        Args:
            sample_rate: sample rate, Hz
            frame_len: duration of signal frame, seconds
            frame_overlap: frame overlap
            offset: decoder offset
            device: 'cpu' or 'cuda' torch inference device
        '''
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.frame_overlap = frame_overlap
        self.offset = offset
        self.device = device

    def load_model(self):
        self.frame_asr = create_quartznet_streaming_asr(
            sample_rate = self.sample_rate,
            frame_len = self.frame_len,
            frame_overlap = self.frame_overlap,
            offset = self.offset,
            device = self.device)

    async def capture_and_transcribe(self,
                                     stream_obj,
                                     started_future: asyncio.Future = None,
                                     loop = None):
        if loop is None:
            loop = asyncio.get_running_loop()
        async for block, status in stream_obj.generator(started_future):
            process_func = functools.partial(self.frame_asr.transcribe, inputs=block)
            transcriptions = await loop.run_in_executor(None, process_func)
            yield transcriptions

    async def transcribe(self, input, loop = None):
        process_func = functools.partial(self.frame_asr.transcribe, inputs=input)
        return await loop.run_in_executor(None, process_func)
    