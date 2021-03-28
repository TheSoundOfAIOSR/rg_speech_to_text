from rgws.interface import WebsocketClient
import json, logging, asyncio


class SimpleClientInterface(WebsocketClient):
    def __init__(self, **kwargs):
        super(SimpleClientInterface, self).__init__(**kwargs)

    """
    This is business logic for client, basically in this example
    we just connects to server and trying to call `example_func` once
    then exits.
    """
    async def _producer(self, websocket):
        logging.debug(await self.status())
        logging.debug(await self.setup_model())
        logging.debug(await self.status())
        logging.debug(await self.start('Mic 1'))
        logging.debug(await self.status())
        logging.debug(await self.stop())
        logging.debug(await self.status())
        await asyncio.sleep(1)