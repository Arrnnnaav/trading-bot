from abc import ABC, abstractmethod


class BrokerBase(ABC):
    @abstractmethod
    def get_price(self, ticker: str) -> float: ...

    @abstractmethod
    def place_order(self, ticker: str, side: str, size_inr: float) -> dict: ...

    @abstractmethod
    def close_position(
        self, broker_order_id: str, ticker: str, side: str, quantity: float = 0.0
    ) -> dict: ...

    @abstractmethod
    def get_ohlcv(self, ticker: str, interval: str, limit: int) -> list[dict]: ...
