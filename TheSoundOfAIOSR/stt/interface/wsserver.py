from rgws.interface import WebsocketServer
import json, logging, asyncio

from TheSoundOfAIOSR.stt.control.stt_sm import SpeechToTextSM

class SimpleServerInterface(WebsocketServer):
    def __init__(self, stt, **kwargs):
        super(SimpleServerInterface, self).__init__(**kwargs)
        self._stt = stt
        self._register(self.setup_model)
        self._register(self.status)
        self._register(self.start)
        self._register(self.stop)
        self._control = SpeechToTextSM(state=SpeechToTextSM.States.uninitialized)

    """
    This overrides _consumer method in WebsocketServer, there should
    business logic be placed if any. At this point we are just 
    dispatching function from message and sending result back.
    """
    async def _consumer(self, ws, message):
        ret = await self.dispatch(message)
        async for gen in ret:
            await ws.send_json(gen)

    
    async def setup_model(self):
        try:
            successful = await self._control.load_stt_models(
                    self._stt, logger=logging, timeout_sec = 10.0)
            if successful: 
                yield {"resp": True}
            else:
                yield {"resp": False, "error": "STT model loading timed out!"}
        except RuntimeError as re:
            yield {"resp": False, "error": "{0}".format(re)}

    
    async def status(self):
        yield {"resp": f"{self._control.state.name}"
                         if self._control.state else 'None'}

    
    async def start(self, name):
        """ start capturing with microphone device given in `name` param.
        """
        try:
            yield {"resp":
                await self._control.start_capture_and_transcribe(
                        self._stt, source_device=name, logger=logging)}
        except RuntimeError as re:
            yield {"reps": False, "error": "{0}".format(re)}
    

    async def stop(self):
        try:
            yield {"resp":
                await self._control.stop_transcription(self._stt, logger=logging)}
        except RuntimeError as re:
            yield {"reps": False, "error": "{0}".format(re)}
