from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import datetime


from environment.env import (
    OwnTrade,
    Real_Data_Env,
    MarketEvent,
    Order,
    update_best_positions,
)


class BestPosStrategy:
    """
    This strategy places ask and bid order at the best position every `delay` nanoseconds.
    If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(
        self,
        delay: float,
        hold_time: Optional[float] = None,
        initial_position: Optional[float] = None,
        min_position: Optional[float] = None,
        max_position: Optional[float] = None,
        trade_size: float = 0.01,
        maker_fee: float = -0.00004,
        log: bool = True,
    ) -> None:
        """
        Args:
            delay(float): delay between orders in nanoseconds
            hold_time(Optional[float]): holding time in nanoseconds
        """
        self.trade_size = trade_size
        self.maker_fee = maker_fee

        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, 5e-2)
        self.hold_time = hold_time

        if initial_position is None:
            initial_position = (max_position + min_position) / 2
        self.min_position = min_position
        self.max_position = max_position
        self.inventory = initial_position

        self.model = "BestPosStrategy"

        self.actions_history = []

        self.log = log

    def run(self, sim: Real_Data_Env, count: int = 100000) -> Tuple[
        List[OwnTrade],
        List[MarketEvent],
        List[Union[OwnTrade, MarketEvent]],
        List[Order],
    ]:
        """
        This function runs simulation

        Args:
            sim(Real_Data_Env): simulator
            count(int): number of iterations
        Returns:
            trades_list(List[OwnTrade]): list of our executed trades
            md_list(List[MarketEvent]): list of market data received by strategy
            orders_list(List[dict]): list of all orders placed by strategy
            updates_list( List[ Union[OwnTrade, MarketEvent] ] ): list of all updates
            received by strategy(market data and information about executed trades)
        """

        # market data list
        market_event_list: List[MarketEvent] = []
        # executed trades list
        trades_list: List[OwnTrade] = []
        # all updates list
        updates_list = []
        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}

        t1 = datetime.datetime.now().timestamp()
        t2 = t1

        tick = 0

        while tick < count:
            t2 = datetime.datetime.now().timestamp()

            # get update from simulator
            receive_ts, updates = sim.tick()
            tick += 1

            if updates is None:
                break

            if self.log and tick % 50000 == 0:
                simulated_time = (
                    receive_ts
                    - datetime.datetime(year=2022, month=10, day=1, hour=2).timestamp()
                )
                print(
                    " " * 200,
                    end="\r",
                )
                print(
                    f"Elapsed time: {t2 - t1:.2f}s",
                    f"Simulated time: {simulated_time//3600:.2f}h {(simulated_time%3600)//60:.2f}m {simulated_time%60:.2f}s",
                    end="\r",
                )

            # save updates
            updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MarketEvent):
                    best_bid, best_ask = update_best_positions(
                        best_bid, best_ask, update
                    )
                    market_event_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)

                    if update.side == "buy":
                        self.inventory += update.size
                    else:
                        self.inventory -= update.size
                else:
                    assert False, "invalid type of update!"

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts

                # place order
                if self.max_position is not None and self.inventory > self.max_position:
                    ask_order = sim.place_order(
                        receive_ts, self.trade_size, "sell", best_ask
                    )
                    ongoing_orders[ask_order.order_id] = ask_order
                    self.actions_history.append(
                        (
                            receive_ts,
                            self.inventory,
                            "(0,0)",
                            best_ask,
                            -np.inf,
                        )
                    )
                elif (
                    self.min_position is not None and self.inventory < self.min_position
                ):
                    bid_order = sim.place_order(
                        receive_ts, self.trade_size, "buy", best_bid
                    )
                    ongoing_orders[bid_order.order_id] = bid_order
                    self.actions_history.append(
                        (
                            receive_ts,
                            self.inventory,
                            "(0,0)",
                            np.inf,
                            best_bid,
                        )
                    )
                else:
                    bid_order = sim.place_order(
                        receive_ts, self.trade_size, "buy", best_bid
                    )
                    ask_order = sim.place_order(
                        receive_ts, self.trade_size, "sell", best_ask
                    )
                    ongoing_orders[bid_order.order_id] = bid_order
                    ongoing_orders[ask_order.order_id] = ask_order
                    self.actions_history.append(
                        (
                            receive_ts,
                            self.inventory,
                            "(0,0)",
                            best_ask,
                            best_bid,
                        )
                    )

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        if self.log:
            print(f"Simulation runned for {t2 - t1:.2f}s", " " * 50)

        return trades_list, market_event_list, self.actions_history, updates_list

    def reset(self):
        self.inventory = (self.max_position + self.min_position) / 2
        self.actions_history = []


class StoikovStrategy:
    """
    This strategy places ask and bid order every `delay` nanoseconds.
    If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(
        self,
        delay: float,
        initial_position: Optional[float] = 0,
        hold_time: Optional[float] = None,
        trade_size: Optional[float] = 0.01,
        risk_aversion: Optional[float] = 0.5,
        k: Optional[float] = 1.5,
        maker_fee: Optional[float] = -0.00004,
        log: bool = True,
    ) -> None:
        """
        Args:
            delay(float): delay between orders in nanoseconds
            hold_time(Optional[float]): holding time in nanoseconds
        """
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, 5e-2)
        self.hold_time = hold_time
        self.order_size = trade_size
        self.last_mid_prices = []

        self.inventory = initial_position
        self.maker_fee = maker_fee

        self.gamma = risk_aversion
        self.k = k
        self.current_bid_order_id = None
        self.current_ask_order_id = None
        self.previous_bid_order_id = None
        self.previous_ask_order_id = None
        self.actions_history = []

        self.model = "Stoikov"
        self.log = log

    def run(self, sim: Real_Data_Env, count: int = 10000) -> Tuple[
        List[OwnTrade],
        List[MarketEvent],
        List[dict],
        List[Union[OwnTrade, MarketEvent]],
    ]:
        """
        This function runs simulation

        Args:
            sim(Real_Data_Env): simulator
        Returns:
            trades_list(List[OwnTrade]): list of our executed trades
            md_list(List[MarketEvent]): list of market data received by strategy
            updates_list( List[ Union[OwnTrade, MarketEvent] ] ): list of all updates
            received by strategy(market data and information about executed trades)
            all_orders(List[Orted]): list of all placed orders
        """

        # market data list
        market_event_list: List[MarketEvent] = []
        # executed trades list
        trades_list: List[OwnTrade] = []
        # all updates list
        updates_list = []
        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []

        t1 = datetime.datetime.now().timestamp()
        t2 = t1

        tick = 0
        while tick < count:
            t2 = datetime.datetime.now().timestamp()

            # get update from simulator
            receive_ts, updates = sim.tick()
            tick += 1

            if updates is None:
                break

            if self.log and tick % 50000 == 0:
                simulated_time = (
                    receive_ts
                    - datetime.datetime(year=2022, month=10, day=1, hour=2).timestamp()
                )
                print(
                    " " * 200,
                    end="\r",
                )
                print(
                    f"Elapsed time: {t2 - t1:.2f}s",
                    f"Simulated time: {simulated_time//3600:.2f}h {(simulated_time%3600)//60:.2f}m {simulated_time%60:.2f}s",
                    end="\r",
                )

            # save updates
            updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MarketEvent):
                    best_bid, best_ask = update_best_positions(
                        best_bid, best_ask, update
                    )
                    mid_price = (best_bid + best_ask) / 2
                    if len(self.last_mid_prices) < 500:
                        self.last_mid_prices.append(mid_price)
                    else:
                        self.last_mid_prices.append(mid_price)
                        self.last_mid_prices.pop(0)
                    market_event_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    if update.execute == "TRADE":
                        if update.side == "sell":
                            self.inventory -= update.size
                        elif update.side == "buy":
                            self.inventory += update.size

                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else:
                    assert False, "invalid type of update!"

            if receive_ts - prev_time >= self.delay and len(ongoing_orders) == 0:
                prev_time = receive_ts
                """
                reservation_price = s - q * gamma * (sigma**2) * (T - t)
                delta_bid and delta_ask are equivalently distant from the reservation_orice
                delta_bid + delta_ask = gamma * (sigma**2) * (T-t) + 2/gamma * ln(1 + gamma/k)
                k = K*alpha
                
                s      : current mid_price
                q      : current position in asset
                sigma  : parameter in the Geometric Brownian Motion equation [dS_t = sigma dw_t]
                T      : termination time
                t      : current time
                gamma  : risk-aversion parameter of the optimizing agents (across the economy)
                K      : higher K means that market order volumes have higher impact on best price changes
                alpha  : higher alpha means higher probability fo large market orders
                
                """
                if len(self.last_mid_prices) == 500:
                    sigma = np.std(
                        self.last_mid_prices
                    )  ## per update --> need to scale it to the "per second" terminology
                else:
                    sigma = 1
                sigma = sigma * np.sqrt(1 / 0.032)
                delta_t = 0.1  ## there is approximately 0.1 seconds in between the orderbook uprates (nanoseconds / 1e9 = seconds)
                q = self.inventory

                reservation_price = mid_price - q * self.gamma * (sigma**2) * delta_t
                deltas_ = self.gamma * (sigma**2) * delta_t + 2 / self.gamma * np.log(
                    1 + self.gamma / self.k
                )
                bid_price = np.round(reservation_price - deltas_ / 2, 1)
                ask_price = np.round(reservation_price + deltas_ / 2, 1)

                bid_order = sim.place_order(
                    receive_ts, self.order_size, "buy", bid_price
                )
                ask_order = sim.place_order(
                    receive_ts, self.order_size, "sell", ask_price
                )

                self.actions_history.append(
                    (
                        receive_ts,
                        self.inventory,
                        f"({bid_price},{ask_price})",
                        best_ask,
                        best_bid,
                    )
                )

                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                self.previous_bid_order_id = self.current_bid_order_id
                self.previous_ask_order_id = self.current_ask_order_id

                self.current_bid_order_id = bid_order.order_id
                self.current_ask_order_id = ask_order.order_id

                all_orders += [bid_order, ask_order]

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            if self.previous_bid_order_id in ongoing_orders.keys():
                sim.cancel_order(receive_ts, self.previous_bid_order_id)
                to_cancel.append(self.previous_bid_order_id)
            if self.previous_ask_order_id in ongoing_orders.keys():
                sim.cancel_order(receive_ts, self.previous_ask_order_id)
                to_cancel.append(self.previous_ask_order_id)
            for ID in to_cancel:
                try:
                    ongoing_orders.pop(ID)
                except:
                    continue

        if self.log:
            print(f"Simulation runned for {t2 - t1:.2f}s", " " * 50)

        return trades_list, market_event_list, self.actions_history, updates_list

    def reset(self):
        self.inventory = 0
        self.last_mid_prices = []
        self.current_bid_order_id = None
        self.current_ask_order_id = None
        self.previous_bid_order_id = None
        self.previous_ask_order_id = None
