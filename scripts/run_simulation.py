#!/usr/bin/env python3
"""Command line interface for HistoricalSimulator."""

from __future__ import annotations

import argparse
import asyncio
import pandas as pd

if __package__ is None or __package__ == "":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bot.config import load_config
from bot.data_handler import DataHandler
from bot.model_builder import ModelBuilder
from bot.trade_manager import TradeManager
from bot.simulation import HistoricalSimulator
from bot.utils import logger


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run historical simulation")
    parser.add_argument("--start", required=True, help="start timestamp YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="end timestamp YYYY-MM-DD")
    parser.add_argument("--speed", type=float, default=60.0, help="time acceleration factor")
    args = parser.parse_args()

    start_ts = pd.to_datetime(args.start, utc=True)
    end_ts = pd.to_datetime(args.end, utc=True)

    cfg = load_config("config.json")
    dh = DataHandler(cfg, None, None)
    try:
        await dh.load_initial()
    except RuntimeError as exc:
        logger.error("Failed to load initial data: %s", exc)
        return
    mb = ModelBuilder(cfg, dh, None)
    tm = TradeManager(cfg, dh, mb, None, None)
    sim = HistoricalSimulator(dh, tm)
    await sim.run(start_ts, end_ts, args.speed)

if __name__ == "__main__":
    asyncio.run(main())
