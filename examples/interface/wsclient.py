from rgws.interface import WebsocketClient
import json, logging, asyncio


class SimpleClientInterface(WebsocketClient):
    def __init__(self, **kwargs):
        super(SimpleClientInterface, self).__init__(**kwargs)

    """
    This simulates a producer cycle consisting from setup_mode 
    and two recordings.
    """
    async def _producer(self, websocket):
        logging.debug("request status")
        r = await self.status()
        logging.debug(r)
        
        if r['resp'] == 'uninitialized':
            # we try recording, but should not allow us    
            logging.debug("request start 1")
            r = await self.start('1')
            assert r['resp'] == False
            assert r['error'] == 'event trigger failed: the "start_capture_and_transcribe" event cannot be triggered from the current state of "uninitialized"'
            logging.debug("start correctly refused")
            # we initialize
            logging.debug("request setup_model")
            r = await self.setup_model()
            logging.debug(r)
            assert r['resp'] == True

        logging.debug("request status")
        r = await self.status()
        logging.debug(r)
        if r['resp'] == 'capture_and_transcribe':
            # in case there is a transcription in progress, stop
            logging.debug("request stop")
            r = await self.stop()
            assert r['resp'] == True

        # start capture and transcribe
        logging.debug("request start 1")
        r = await self.start('1')
        logging.debug(r)
        assert r['resp'] == True

        # wait for speech
        await asyncio.sleep(5)
        
        # setup_model is not allowed during live capture
        logging.debug("request setup_model")
        r = await self.setup_model()
        assert r['resp'] == False
        assert r['error'] == 'event trigger failed: the "load_stt_models" event cannot be triggered from the current state of "capture_and_transcribe"'
        logging.debug("setup_model correctly refused")

        # wait for speech 5 more
        await asyncio.sleep(5)

        # stop transcription
        logging.debug("request stop")
        r = await self.stop()
        logging.debug("transcribed text is: %s", r['resp'])

        # after stop, we should be in models_ready status
        logging.debug("request status")
        r = await self.status()
        logging.debug(r)
        assert r['resp'] == 'models_ready'

        # start capture and transcribe
        logging.debug("request start 1")
        r = await self.start('1')
        logging.debug(r)
        assert r['resp'] == True

        # wait for speech
        await asyncio.sleep(5)

        # attempt a second capture and transcribe in parallel
        logging.debug("request start 1")
        r = await self.start('1')
        assert r['resp'] == False
        assert r['error'] == 'event trigger failed: the "start_capture_and_transcribe" event cannot be triggered from the current state of "capture_and_transcribe"'
        logging.debug("second capture attempt correctly refused")

        # wait for speech
        await asyncio.sleep(5)

        # stop transcription
        logging.debug("request stop")
        r = await self.stop()
        logging.debug("transcribed text is: %s", r['resp'])

        # after stop, we should be in models_ready status
        logging.debug("request status")
        r = await self.status()
        logging.debug(r)
        assert r['resp'] == 'models_ready'

        # start capture and transcribe second time
        logging.debug("request start 1")
        r = await self.start('1')
        logging.debug(r)
        assert r['resp'] == True

        # wait for speech
        await asyncio.sleep(10)

        # stop transcription
        logging.debug("request stop")
        r = await self.stop()
        logging.debug("transcribed text is: %s", r['resp'])

        # after stop, we should be in models_ready status
        logging.debug("request status")
        r = await self.status()
        logging.debug(r)
        assert r['resp'] == 'models_ready'

        logging.debug('Test passed.')
        await asyncio.sleep(1)