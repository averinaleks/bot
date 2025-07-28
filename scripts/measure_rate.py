#!/usr/bin/env python3
import os, sys; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import asyncio
import json
from contextlib import suppress
import pandas as pd
from config import BotConfig
from data_handler import DataHandler

class DummyExchange:
    pass

async def measure(n=1000, seconds=1.0):
    cfg = BotConfig(cache_dir='/tmp')
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
