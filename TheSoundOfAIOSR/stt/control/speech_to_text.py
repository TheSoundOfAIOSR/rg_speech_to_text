import asyncio
import functools
import logging


from TheSoundOfAIOSR.stt.wavenet.audiointerface.interface import MicrophoneCaptureFailed
from TheSoundOfAIOSR.stt.wavenet.audiointerface.interface import MicrophoneStreaming


class SpeechToText:
    def __init__(self, asr, sample_rate: int,
                 frame_len: float, frame_overlap: int, 
                 decoder_offset=0, channels=1):
        """ 
        Args:
            asr - ASR engine
            sample_rate - sample rate, Hz
            frame_len - duration of signal frame, seconds
            frame_overlap - frame overlap (for example 2)
            decoder_offset - decoder offset
            channels - number of audio channels (expect mono signal)  
        """
        self._asr = asr
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_len = frame_len
        self._frame_overlap = frame_overlap
        self._decoder_offset = decoder_offset
        self._block_size = int(frame_len * sample_rate * frame_overlap)
        self._transcription_queue = asyncio.Queue()
        self._loop = None
        self._asr_task = None


    def _ensure_loop(self):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return self._loop


    def load_model(self, logger: logging.Logger):
        self._asr.load_model()


    async def transcribe_audio_file(self, audio_file_name: str, logger: logging.Logger):
        ...        


    async def start_capture_and_transcribe(
            self, sound_device: str, logger: logging.Logger):
        logger.debug("start_capture_and_transcribe on %s", sound_device)
        loop = self._ensure_loop()
        start_time = loop.time()
        started_future = loop.create_future()
        self._asr_task = asyncio.create_task(
            self._capture_and_transcribe(sound_device, started_future, loop, logger))
        # we are waiting the result of starting the sound device capture generator
        # for the outcome, which we return
        started = await started_future
        # from this state the capture and transcribe generator coroutine continue its job
        delta_time = loop.time() - start_time
        if started:
            logger.debug("starting capture on %s took %s", sound_device, delta_time)
        else:
            logger.warn("starting capture on %s failed in %s", sound_device, delta_time)
        return started


    async def stop_transcription(self, logger:logging.Logger):
        def _cancel_task(task, future):
            task.cancel()
            future.set_result(None)

        loop = self._ensure_loop()
        start_time = loop.time()
        closed_future = loop.create_future()
        # schedule a threasafe cancel stask (when transcription coroutine is suspended)
        loop.call_soon_threadsafe(_cancel_task, self._asr_task, closed_future)
        stopped = await closed_future
        delta_closing_time = loop.time() - start_time
        # now we can fetch all the transcription output
        full_transcription = await asyncio.create_task(self._fetch_all_transcription(logger))
        delta_time = loop.time() - start_time
        logger.debug("stopping transcription took %s, and with text fetch took %s",
                     delta_closing_time, delta_time)
        return full_transcription


    async def _fetch_all_transcription(self, logger: logging.Logger):
        fulltext = ""
        try:
            while True:
                fragment = self._transcription_queue.get_nowait()
                fulltext += fragment + ' '
        except asyncio.QueueEmpty:
            logger.debug("transcribed text is: %s", fulltext)
            return fulltext


    async def _capture_and_transcribe(self,
                                      sound_device: str,
                                      started_future: asyncio.Future,
                                      loop,
                                      logger: logging.Logger):
        try:
            stream = MicrophoneStreaming(buffersize=self._block_size, loop=loop, interface="sd") \
                    .stream(logger=logger)
            # consuming the audio capture stream and online transcription
            async for transcribed in self._asr.capture_and_transcribe(
                        stream, started_future, loop=loop):
                if not transcribed == "":
                    try:
                        self._transcription_queue.put_nowait(transcribed)
                    except asyncio.QueueFull as fullErr:
                        logger.warning("The transcription queue is too full.")
                        raise fullErr
        except MicrophoneCaptureFailed as captureErr:
            logger.error("Unable to capture from {0}", sound_device)
            raise captureErr
        except RuntimeError as rerr:
            logger.error("RuntimeError in capture and transcribe")
            raise rerr

