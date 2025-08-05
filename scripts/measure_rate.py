#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from contextlib import suppress
import pandas as pd
import tempfile

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bot.config import BotConfig
from bot.data_handler import DataHandler

class DummyExchange:
    pass

async def measure(n=1000, seconds=1.0):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = BotConfig(cache_dir=tmpdir)
        dh = DataHandler(cfg, None, None, exchange=DummyExchange())
        ts = int(pd.Timestamp.now(tz='UTC').timestamp()*1000)
        msg = json.dumps({
            'topic': 'kline.1.BTCUSDT',
            'data': [{
                'start': ts,
            'open': 1,
            'high': 1,
            'low': 1,
            'close': 1,
            'volume': 1
        }]
    })
        for _ in range(n):
            await dh.ws_queue.put((1, (['BTCUSDT'], msg, 'primary')))
        task = asyncio.create_task(dh._process_ws_queue())
        await asyncio.sleep(seconds)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        rate = len(dh.process_rate_timestamps) / dh.process_rate_window
        print('rate', rate)

if __name__ == '__main__':
    asyncio.run(measure())
