#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

import asyncio

from TheSoundOfAIOSR.stt.interface.wsserver import SimpleServerInterface

if __name__ == "__main__":
    srv = SimpleServerInterface(host="localhost", port=8786)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(srv.run())
    loop.run_forever()