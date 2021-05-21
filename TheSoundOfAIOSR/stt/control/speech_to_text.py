import asyncio
import functools
import logging
import numpy


from TheSoundOfAIOSR.audiointerface.capture import MicrophoneCaptureFailed
from TheSoundOfAIOSR.audiointerface.capture import MicrophoneStreaming

logger = logging.getLogger('sptt')

class SpeechToText:
    def __init__(self, asr, block_size: int, offline_mode=False):
        """ 
        Args:
            asr: ASR engine
            block_size: block size
            offline_mode: first capture all, then transcribe at stop
        """
        self._asr = asr
        self._offline_mode = offline_mode
        self._block_size = block_size
        self._buffer_queue = asyncio.Queue()
        self._loop = None
        self._asr_task = None


    def _ensure_loop(self):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return self._loop


    def load_model(self):
        self._asr.load_model()


    async def start_capture_and_transcribe(
            self, sound_device: str):
        loop = self._ensure_loop()
        start_time = loop.time()
        started_future = loop.create_future()
        
        if self._offline_mode:
            logger.debug("start capturing in  offline mode from %s", sound_device)
            self._asr_task = asyncio.create_task(
                self._capture_offline(sound_device, started_future, loop))
        else:
            logger.debug("start capture and transcribe online from %s", sound_device)
            self._asr_task = asyncio.create_task(
                self._capture_and_transcribe(sound_device, started_future, loop))
        
        # we are waiting the result of starting the sound device capture generator
        # for the outcome, which we return
        started = await started_future
        # from this state the capture and transcribe generator coroutine continue its job
        delta_time = loop.time() - start_time
        if started:
            logger.debug("capturing on %s took %s", sound_device, delta_time)
        else:
            logger.warn("capturing on %s failed in %s", sound_device, delta_time)
        return started


    async def stop_transcription(self):
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
        
        if self._offline_mode:
            full_transcription = await asyncio.create_task(
                    self._asr.transcribe(await self._fetch_all_audio(), loop))
        else:
            full_transcription = await asyncio.create_task(
                    self._fetch_all_transcription())
        
        delta_time = loop.time() - start_time
        logger.debug("stopping transcription took %s, and with text fetch took %s",
                     delta_closing_time, delta_time)
        return full_transcription


    async def _fetch_all_transcription(self):
        fulltext = ""
        try:
            while True:
                fragment = self._buffer_queue.get_nowait()
                fulltext += fragment + ' '
        except asyncio.QueueEmpty:
            logger.debug("transcribed text is: %s", fulltext)
            return fulltext


    async def _capture_and_transcribe(self,
                                      sound_device: str,
                                      started_future: asyncio.Future,
                                      loop):
        try:
            #logger.debug("_capture_and_transcribe ...... ")
            stream = MicrophoneStreaming(buffersize=self._block_size, loop=loop)
            #logger.debug("MicrophoneStreaming created ")
            # consuming the audio capture stream and online transcription
            async for transcribed in self._asr.capture_and_transcribe(
                        stream, started_future, loop=loop):
                #logger.print(transcribed)
                if not transcribed == "":
                    try:
                        self._buffer_queue.put_nowait(transcribed)
                    except asyncio.QueueFull as fullErr:
                        logger.warning("The transcription queue is too full.")
                        raise fullErr
        except MicrophoneCaptureFailed as captureErr:
            logger.error("Unable to capture from {0}", sound_device)
            raise captureErr
        except RuntimeError as rerr:
            logger.error("RuntimeError in capture and transcribe")
            raise rerr


    async def _capture_offline(self, 
                               sound_device: str,
                               started_future: asyncio.Future,
                               loop):
        try:
            stream = MicrophoneStreaming(buffersize=self._block_size, loop=loop)
            async for block, status in stream.generator(started_future):
                try:
                    self._buffer_queue.put_nowait(block)
                except asyncio.QueueFull as fullErr:
                    logger.warning("The audio buffer queue is too full.")
                    raise fullErr
        except MicrophoneCaptureFailed as captureErr:
            logger.error("Unable to capture from {0}", sound_device)
            raise captureErr
        except RuntimeError as rerr:
            logger.error("RuntimeError in capture and transcribe")
            raise rerr


    async def _fetch_all_audio(self):
        list_of_blocks = []
        try:
            while True:
                list_of_blocks.append(self._buffer_queue.get_nowait())
        except asyncio.QueueEmpty:
            ...
        return numpy.concatenate(list_of_blocks, axis=0)

