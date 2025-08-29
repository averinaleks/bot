"""In-memory storage helpers for data handler."""
from dataclasses import dataclass, field
from typing import Dict

DEFAULT_PRICE = 0.0


@dataclass
class PriceStorage:
    prices: Dict[str, float] = field(default_factory=dict)

    def get(self, symbol: str):
        return self.prices.get(symbol)

    def set(self, symbol: str, price: float) -> None:
        self.prices[symbol] = price


price_storage = PriceStorage()
