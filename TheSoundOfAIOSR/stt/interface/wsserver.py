from rgws.interface import WebsocketServer
import json, logging, asyncio

from TheSoundOfAIOSR.stt.control.stt_sm import SpeechToTextSM

class SimpleServerInterface(WebsocketServer):
    def __init__(self, **kwargs):
        super(SimpleServerInterface, self).__init__(**kwargs)
        self._register(self.setup_model)
        self._register(self.status)
        self._register(self.select_microphone)
        self._register(self.start)
        self._register(self.stop)
        self._register(self.fetch)
        self._control = SpeechToTextSM()

    """
    This overrides _consumer method in WebsocketServer, there should
    business logic be placed if any. At this point we are just 
    dispatching function from message and sending result back.
    """
    async def _consumer(self, websocket, message):
        ret = await self.dispatch(message)
        async for gen in ret:
            await websocket.send(gen)

    async def setup_model(self):
        await self._control.reset(logger=logging)
        await self._control.load_stt_models(logger=logging)
        yield json.dumps({"resp": True})

    async def status(self):
        yield json.dumps({"resp": f"{self._control.state.name if self._control.state else 'None'}"})

    async def select_microphone(self, name):
        logging.debug("select microphone {}", name)
        self._control.select_microphone(name)
        yield json.dumps({"resp": name})

    async def start(self):
        await self._control.start_capture_and_transcribe(logger=logging)
        yield json.dumps({"resp": True})

    async def stop(self):
        await self._control.stop_transcription(logger=logging)
        yield json.dumps({"resp": True})

    async def fetch(self):
        logging.debug("fetch")
        yield json.dumps({"resp": "the transcribed test is this"})
