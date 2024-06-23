import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from sortedcontainers import SortedDict
from typing import List, Optional, Tuple, Union, Deque, Dict


from scipy.linalg import expm


@dataclass
class Order:
    order_id: str
    place_ts: float
    exchange_ts: float
    side: str
    price: float
    size: float


@dataclass
class CancelOrder:
    exchange_ts: float
    id_to_delete: int


@dataclass
class Trade:
    exchange_ts: float
    receive_ts: float
    side: str
    price: float
    quantity: float


@dataclass
class OrderBook:
    exchange_ts: float
    receive_ts: float
    asks: List[Tuple[float, float]]  # tuple[price, size]
    bids: List[Tuple[float, float]]  # tuple[price, size]


@dataclass
class OwnTrade:
    placing_ts: float
    exchange_ts: float
    receive_ts: float
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float
    execute: str  # BOOK or TRADE


@dataclass
class MarketEvent:
    exchange_ts: float
    receive_ts: float
    orderbook: Optional[OrderBook] = None
    trade: Optional[Trade] = None


def update_best_positions(
    best_bid, best_ask, market_event: MarketEvent, levels: bool = False
) -> Tuple[float, float]:
    """
    Update best bid and best ask prices

    Args:
        best_bid(float): best bid price
        best_ask(float): best ask price
        market_event(MarketEvent): market data
        levels(bool): return  levels or not

    Returns:
        best_bid(float): best bid price
        best_ask(float): best ask price
        asks_price(List[float]): ask price levels
        bids_price(List[float]): bid price levels
        asks_volume(List[float]): ask volume levels
        bids_volume(List[float]): bid volume levels

    """

    if market_event.trade is None:  # MarketEvent is OrderBook
        best_bid = market_event.orderbook.bids[0][0]
        best_ask = market_event.orderbook.asks[0][0]

        # return ask and bid levels
        if levels:
            asks_price = [level[0] for level in market_event.orderbook.asks]
            bids_price = [level[0] for level in market_event.orderbook.bids]

            asks_volume = [level[0] for level in market_event.orderbook.asks]
            bids_volume = [level[0] for level in market_event.orderbook.bids]

            return best_bid, best_ask, asks_price, bids_price, asks_volume, bids_volume
        else:
            return best_bid, best_ask

    else:  # MarketEvent is Trade
        if market_event.trade.side == "buy":
            best_ask = max(market_event.trade.price, best_ask)
        elif market_event.trade.side == "sell":
            best_bid = min(best_bid, market_event.trade.price)
        return best_bid, best_ask


class Real_Data_Env:
    def __init__(
        self,
        market_data: List[MarketEvent],
        execution_latency: float,
        market_event_latency: float,
    ) -> None:
        """
        Args:
            market_data(List[MarketEvent]): market data
            execution_latency(float): latency in nanoseconds for order execution
            market_event_latency(float): latency in nanoseconds for market event reception
        """

        self.price_history = []

        # market event queue
        self.market_event_queue = deque(market_data)
        # agent action queue
        self.actions_queue: Deque[Union[Order, CancelOrder]] = deque()
        # all the event that will be sent to the strategy : SordetDict: receive_ts -> [updates]
        self.strategy_updates_queue = SortedDict()
        # map : order_id -> Order
        self.ready_to_execute_orders: Dict[int, Order] = {}

        # current market_event
        self.market_event: Optional[MarketEvent] = None

        # current ids
        self.order_id = 0
        self.trade_id = 0
        # current bid and ask
        self.best_bid = -np.inf
        self.best_ask = np.inf
        # current trade
        self.trade_price = {}
        self.trade_price["buy"] = -np.inf
        self.trade_price["sell"] = np.inf
        # last order
        self.last_order: Optional[Order] = None

        # latency
        self.latency = execution_latency
        self.market_event_latency = market_event_latency

    def get_market_event_queue_event_time(self) -> float:
        return (
            np.inf
            if len(self.market_event_queue) == 0
            else self.market_event_queue[0].exchange_ts
        )

    def get_actions_queue_event_time(self) -> float:
        return (
            np.inf
            if len(self.actions_queue) == 0
            else self.actions_queue[0].exchange_ts
        )

    def get_strategy_updates_queue_event_time(self) -> float:
        return (
            np.inf
            if len(self.strategy_updates_queue) == 0
            else self.strategy_updates_queue.keys()[0]
        )

    def get_order_id(self) -> int:
        res = self.order_id
        self.order_id += 1
        return res

    def get_trade_id(self) -> int:
        res = self.trade_id
        self.trade_id += 1
        return res

    def update_best_pos(self) -> None:
        assert not self.market_event is None, "no current market data!"

        self.best_bid = self.market_event.orderbook.bids[0][0]
        self.best_ask = self.market_event.orderbook.asks[0][0]

        self.price_history.append((self.best_bid + self.best_ask) / 2)

    def update_last_trade(self) -> None:
        assert not self.market_event is None, "no current market data!"

        self.trade_price[self.market_event.trade.side] = self.market_event.trade.price

    def delete_last_trade(self) -> None:
        self.trade_price["buy"] = -np.inf
        self.trade_price["sell"] = np.inf

    def update_market_event(self, market_event: MarketEvent) -> None:

        # current orderbook
        self.market_event = market_event

        # update the environment
        if self.market_event.orderbook is None:
            self.update_last_trade()
        else:
            self.update_best_pos()

        # add market_event to strategy_updates_queue
        if not market_event.receive_ts in self.strategy_updates_queue.keys():
            self.strategy_updates_queue[market_event.receive_ts] = []
        self.strategy_updates_queue[market_event.receive_ts].append(market_event)

    def update_action(self, action: Union[Order, CancelOrder]) -> None:

        if isinstance(action, Order):
            # save last order to try to execute it aggressively
            self.last_order = action
        elif isinstance(action, CancelOrder):
            # cancel order
            if action.id_to_delete in self.ready_to_execute_orders:
                self.ready_to_execute_orders.pop(action.id_to_delete)
        else:
            assert False, "Wrong action type!"

    def tick(self) -> Tuple[float, List[Union[OwnTrade, MarketEvent]]]:
        """
        Simulation tick

        Returns:
            receive_ts(float): receive timestamp in nanoseconds
            res(List[Union[OwnTrade, MarketEvent]]): simulation result.
        """
        while True:
            # get event time for all the queues
            strategy_updates_queue_et = self.get_strategy_updates_queue_event_time()
            market_event_queue_et = self.get_market_event_queue_event_time()
            actions_queue_et = self.get_actions_queue_event_time()

            # if both queue are empty : no more events
            if market_event_queue_et == np.inf and actions_queue_et == np.inf:
                break

            # strategy queue has minimum event time : execute it
            if strategy_updates_queue_et < min(market_event_queue_et, actions_queue_et):
                break

            if market_event_queue_et <= actions_queue_et:
                self.update_market_event(self.market_event_queue.popleft())
            if actions_queue_et <= market_event_queue_et:
                self.update_action(self.actions_queue.popleft())

            # execute last order aggressively
            self.execute_last_order()
            # execute orders with current orderbook
            self.execute_orders()
            # delete last trade
            self.delete_last_trade()

        # end of simulation
        if len(self.strategy_updates_queue) == 0:
            return np.inf, None
        key = self.strategy_updates_queue.keys()[0]
        res = self.strategy_updates_queue.pop(key)
        return key, res

    def execute_last_order(self) -> None:
        """
        this function tries to execute self.last order aggressively
        """
        # nothing to execute
        if self.last_order is None:
            return

        executed_price, execute = None, None
        #
        if self.last_order.side == "buy" and self.last_order.price >= self.best_ask:
            executed_price = self.best_ask
            execute = "BOOK"
        #
        elif self.last_order.side == "sell" and self.last_order.price <= self.best_bid:
            executed_price = self.best_bid
            execute = "BOOK"

        if executed_price is not None:
            executed_order = OwnTrade(
                self.last_order.place_ts,  # when we place the order
                self.market_event.exchange_ts,  # exchange ts
                self.market_event.exchange_ts + self.market_event_latency,  # receive ts
                self.get_trade_id(),  # trade id
                self.last_order.order_id,
                self.last_order.side,
                self.last_order.size,
                executed_price,
                execute,
            )

            # add order to strategy update queue
            if not executed_order.receive_ts in self.strategy_updates_queue:
                self.strategy_updates_queue[executed_order.receive_ts] = []
            self.strategy_updates_queue[executed_order.receive_ts].append(
                executed_order
            )
        else:
            self.ready_to_execute_orders[self.last_order.order_id] = self.last_order

        # delete last order
        self.last_order = None

    def execute_orders(self) -> None:
        executed_orders_id = []
        for order_id, order in self.ready_to_execute_orders.items():

            executed_price, execute = None, None

            #
            if order.side == "buy" and order.price >= self.best_ask:
                executed_price = order.price
                execute = "BOOK"
            #
            elif order.side == "sell" and order.price <= self.best_bid:
                executed_price = order.price
                execute = "BOOK"
            #
            elif order.side == "buy" and order.price >= self.trade_price["sell"]:
                executed_price = order.price
                execute = "TRADE"
            #
            elif order.side == "sell" and order.price <= self.trade_price["buy"]:
                executed_price = order.price
                execute = "TRADE"

            if not executed_price is None:
                executed_order = OwnTrade(
                    order.place_ts,  # when we place the order
                    self.market_event.exchange_ts,  # exchange ts
                    self.market_event.exchange_ts
                    + self.market_event_latency,  # receive ts
                    self.get_trade_id(),  # trade id
                    order_id,
                    order.side,
                    order.size,
                    executed_price,
                    execute,
                )

                executed_orders_id.append(order_id)

                # add order to strategy update queue
                if not executed_order.receive_ts in self.strategy_updates_queue:
                    self.strategy_updates_queue[executed_order.receive_ts] = []
                self.strategy_updates_queue[executed_order.receive_ts].append(
                    executed_order
                )

        # deleting executed orders
        for k in executed_orders_id:
            self.ready_to_execute_orders.pop(k)

    def place_order(self, ts: float, size: float, side: str, price: float) -> Order:
        order = Order(self.get_order_id(), ts, ts + self.latency, side, price, size)
        self.actions_queue.append(order)
        return order

    def cancel_order(self, ts: float, id_to_delete: int) -> CancelOrder:
        ts += self.latency
        delete_order = CancelOrder(ts, id_to_delete)
        self.actions_queue.append(delete_order)
        return delete_order
