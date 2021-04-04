from typing import Optional, List
import asyncio
import logging
import statesman as sm

from TheSoundOfAIOSR.stt.control.speech_to_text import SpeechToText


logger = logging.getLogger('stsm')


def load_stt_models(stt: SpeechToText):
    logger.debug("pre-loading STT models...")
    stt.load_model()
    logger.debug("STT models has been loaded.")


class SpeechToTextSM(sm.StateMachine):
    """
    SpeechToTextSM is an asyncio based state machine, which
    protects agains simultaneous capture sessions and 
    only allows to capture when the speech to text model
    was already loaded.   
    """
    class States(sm.StateEnum):
        uninitialized = "Uninitialized."
        models_preloading = "Preloading the models..."
        models_ready = "The models are preloaded and ready."
        capture_and_transcribe = "Capture and transcribe..."


    @sm.event(None, States.uninitialized)
    async def reset(self, stt: SpeechToText) -> None:
        logger.info("reset into uninitialized state")


    @sm.event(source=[States.uninitialized, States.models_ready],
              target=States.models_preloading, return_type=bool)
    async def load_stt_models(self,
                              stt: SpeechToText,
                              timeout_sec: int = 15):
        logger.info("preload_models action schedules load stt models ...")
        # since it is a long running process, we execute in threadpool
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        future = loop.run_in_executor(None, load_stt_models, stt)
        try:
            result = await asyncio.wait_for(future, timeout_sec, loop=loop)
            # will be resumed only after the model loading is done or timout
            await self.models_ready()
            delta_time = loop.time() - start_time
            logger.debug("preload_models action took %s", delta_time)
            return True
        except asyncio.TimeoutError:
            delta_time = loop.time() - start_time
            logger.error("preload_models timed out after %s", delta_time)
            return False

    
    @sm.event(source=States.models_preloading, target=States.models_ready)
    async def models_ready(self) -> None:
        logger.info("models_ready.")


    @sm.event(source=States.models_ready,
              target=States.capture_and_transcribe,
              return_type=bool)
    async def start_capture_and_transcribe(self,  stt: SpeechToText, source_device: str):
        return await stt.start_capture_and_transcribe(source_device)


    @sm.event(source=States.capture_and_transcribe,
              target=States.models_ready,
              return_type=object)
    async def stop_transcription(self,  stt: SpeechToText):
        return await stt.stop_transcription()
