from typing import Optional, List
import asyncio
import logging
import statesman as sm
import sys

from TheSoundOfAIOSR.stt.control.model_loader import load_stt_models as load_models


class SpeechToTextSM(sm.StateMachine):
    class States(sm.StateEnum):
        uninitialized = "Uninitialized."
        models_preloading = "Preloading the models..."
        models_ready = "The models are preloaded and ready."
        capture_and_transcribe = "Capture and transcribe..."
        idle = "Idle..."

    @sm.event(None, States.uninitialized)
    async def uninitialized(self, logger: logging.Logger) -> None:
        logger.debug("uninitialized action")
    

    @sm.event(source=States.uninitialized, target=States.models_preloading)
    async def load_stt_models(self, logger: logging.Logger) -> None:
        logger.debug("preload_models action schedules load stt models ...")
        # since it is a long running process, we execute in threadpool
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, load_models, logger),
        # it will be resumed only after the load is done
        logger.debug("preload_models action return.")

    
    @sm.event(source=States.models_preloading, target=States.models_ready)
    async def models_ready(self, logger: logging.Logger) -> None:
        logger.debug("models_ready.")


    @sm.event(source=States.models_ready, target=States.capture_and_transcribe)
    async def start_capture_and_transcribe(self, logger: logging.Logger) -> None:
        logger.debug("start_capture_and_transcribe action.")

    
    @sm.event(source=States.capture_and_transcribe, target=States.idle)
    async def stop_transcription(self, logger: logging.Logger) -> None:
        logger.debug("stop_transcription")