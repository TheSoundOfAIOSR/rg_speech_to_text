#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

import asyncio
import logging

from examples.interface.wsclient import SimpleClientInterface

c = SimpleClientInterface(host="localhost", port=8786)

async def main():
    ...

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(c.run(), main()))