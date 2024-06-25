from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import datetime

from environment.env import (
    OwnTrade,
    Real_Data_Env,
    MarketEvent,
    Order,
    update_best_positions,
)
from utils.features import book_imbalance, RSI, volatility, inventory_ratio
from utils.evaluate import trade_to_dataframe, md_to_dataframe


class QLearning:

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.99,
        alpha_decay: float = 0.9999,
        min_alpha: float = 0.01,
    ):
        self.alpha = alpha
        self.gamma = gamma

        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha

    def initialize(self, state_sizes, n_actions):
        self.q_table = np.zeros(state_sizes + [n_actions])

    def choose_action(self, state, epsilon=0.1):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.q_table.shape[-1])
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action):
        self.update_params()

        self.q_table[state + (action,)] += self.alpha * (
            reward
            + self.gamma
            * self.q_table[next_state + (np.argmax(self.q_table[next_state]),)]
            - self.q_table[state + (action,)]
        )

    def update_params(self):
        self.alpha = max(self.alpha * self.alpha_decay, self.min_alpha)

    def __str__(self):
        return f"QLearning(alpha={self.alpha}_gamma={self.gamma}_disc_alpha={self.alpha_decay}_min_alpha={self.min_alpha})"


class SARSA:

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.99,
        alpha_decay: float = 0.9999,
        min_alpha: float = 0.01,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha

    def initialize(self, state_sizes, n_actions):
        self.q_table = np.zeros(state_sizes + [n_actions])

    def choose_action(self, state, epsilon=0.3):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.q_table.shape[-1])
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action):
        self.update_params()

        self.q_table[state + (action,)] += self.alpha * (
            reward
            + self.gamma * self.q_table[next_state + (next_action,)]
            - self.q_table[state + (action,)]
        )

    def update_params(self):
        self.alpha = max(self.alpha * self.alpha_decay, self.min_alpha)

    def __str__(self):
        return f"SARSA(alpha={self.alpha}_gamma={self.gamma}_disc_alpha={self.alpha_decay}_min_alpha={self.min_alpha})"


class RLStrategy:
    """
    This strategy places ask and bid order coming from the model choice every `delay` nanoseconds.
    If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(
        self,
        model: Union[QLearning],
        min_position: float,
        max_position: float,
        delay: float,
        initial_position: Optional[float] = None,
        hold_time: Optional[float] = None,
        trade_size: float = 0.01,
        maker_fee: float = -0.00004,
        order_book_depth: int = 6,
        log: bool = True,
    ) -> None:
        """
        Args:
            model(Union[QLearning]): RL model
            initial_position(float): initial position size
            min_position(float): minimum position size
            max_position(float): maximum position size
            delay(float): delay beetween orders in nanoseconds
            hold_time(Optional[float]): hold time in nanoseconds
            trade_size(float): trade size
            maker_fee(float): maker fee
            order_book_depth(int): order book depth to be considered for action space
        """

        self.maker_fee = maker_fee
        self.trade_size = trade_size
        self.min_position = min_position
        self.max_position = max_position
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, 5e-2)
        self.hold_time = hold_time

        if initial_position is None:
            initial_position = (self.max_position + self.min_position) / 2
        self.initial_position = initial_position

        self.action_dict = {  # id : (ask_level, bid_level)
            i: x
            for (i, x) in enumerate(
                [
                    (i, j)
                    for i in range(order_book_depth)
                    for j in range(order_book_depth)
                ]
            )
        }

        self.state_space = [  # level, min, max for each feature, bin for extreme values
            (8, 0, 1, True),  # inventory ratio
            (10, -1, 1, False),  # book imbalance (mostly 0 here)
            # (10, 0, 1, False),  # spread #  TODO: Define relevant state space
            (5, 0, 0.3, False),  # volatility
            (10, 0, 100, False),  # rsi
        ]

        self.model = model
        self.log = log

        self.reset()

    def place_order(
        self,
        sim: Real_Data_Env,
        action_id: float,
        receive_ts: float,
        asks_price,
        bids_price,
    ):
        ask_level, bid_level = self.action_dict[action_id]
        ask_order = sim.place_order(
            receive_ts, self.trade_size, "sell", asks_price[ask_level]
        )
        bid_order = sim.place_order(
            receive_ts, self.trade_size, "buy", bids_price[bid_level]
        )

        self.ongoing_orders[bid_order.order_id] = (bid_order, "LIMIT")
        self.ongoing_orders[ask_order.order_id] = (ask_order, "LIMIT")

        self.actions_history.append(
            (
                receive_ts,
                self.inventory,
                str(self.action_dict[action_id]),
                asks_price[ask_level],
                bids_price[bid_level],
            )
        )

    def run(self, sim: Real_Data_Env, mode: str, count=10000) -> Tuple[
        List[OwnTrade],
        List[MarketEvent],
        List[dict],
        List[Union[MarketEvent, OwnTrade]],
    ]:
        """
        This function runs simulation

        Args:
            sim(Real_Data_Env): simulator
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
        bids_price = [-np.inf] * len(self.action_dict)
        asks_price = [np.inf] * len(self.action_dict)
        bids_volume = [0] * len(self.action_dict)
        asks_volume = [0] * len(self.action_dict)

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet

        current_state = self.get_state(best_ask, best_bid, [], [], [])
        prev_state = current_state
        current_action = 0
        prev_action = current_action
        prev_mid_price = 0
        reward = 0

        t1 = datetime.datetime.now().timestamp()
        t2 = t1

        tick = 0
        nb_updates = 0
        while tick < count:
            t2 = datetime.datetime.now().timestamp()

            # get update from simulator
            receive_ts, updates = sim.tick()
            tick += 1

            if updates is None:
                break

            if tick % 50000 == 0 and self.log:
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

                # if update is market data, update best position
                if isinstance(update, MarketEvent):
                    if update.orderbook is not None:
                        prev_mid_price = (best_bid + best_ask) / 2
                        (
                            best_bid,
                            best_ask,
                            asks_price,
                            bids_price,
                            asks_volume,
                            bids_volume,
                        ) = update_best_positions(
                            best_bid, best_ask, update, levels=True
                        )
                    market_event_list.append(update)

                # if update is trade, update position and pnl
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)

                    # delete executed trades from the dict
                    if update.order_id in self.ongoing_orders.keys():
                        _, order_type = self.ongoing_orders.pop(update.order_id)

                    # impact of the trade on the position and pnl
                    if order_type == "LIMIT" and update.execute == "TRADE":
                        if update.side == "buy":
                            self.inventory += update.size
                            self.realized_pnl -= (
                                (1 + self.maker_fee) * update.price * update.size
                            )
                        else:
                            self.inventory -= update.size
                            self.realized_pnl = (
                                (1 - self.maker_fee) * update.price * update.size
                            )
                        self.unrealized_pnl = self.inventory * (
                            (best_ask + best_bid) / (2 * prev_mid_price) - 1
                        )

                        reward = self.realized_pnl + self.unrealized_pnl

                        # penalize the agent for having a position too close to the limits (mean-reverting strategy)
                        reward += (
                            -1e4
                            * (
                                abs(
                                    inventory_ratio(
                                        self.inventory,
                                        self.min_position,
                                        self.max_position,
                                    )
                                    - 0.5
                                )
                            )
                            ** 2
                        )

                        if (
                            self.inventory < self.min_position
                            or self.inventory > self.max_position
                        ):
                            reward = -1e5

                        # update state
                        current_state = self.get_state(
                            best_ask,
                            best_bid,
                            sim.price_history,
                            asks_volume,
                            bids_volume,
                        )

                        prev_action = self.update_model(
                            receive_ts,
                            reward,
                            prev_state,
                            prev_action,
                            current_state,
                            nb_updates,
                            mode,
                        )
                        prev_state = current_state
                        nb_updates += 1

                else:
                    assert False, "invalid type of update!"

            # if the delay has passed, place an order
            if receive_ts - prev_time >= self.delay and (len(self.ongoing_orders) == 0):

                self.place_order(sim, prev_action, receive_ts, asks_price, bids_price)

                prev_time = receive_ts

            to_cancel = []
            for ID, (order, order_type) in self.ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)

                    reward = -40 * self.hold_time / self.delay

                    # update state
                    current_state = self.get_state(
                        best_ask, best_bid, sim.price_history, asks_volume, bids_volume
                    )

                    prev_action = self.update_model(
                        receive_ts,
                        reward,
                        prev_state,
                        prev_action,
                        current_state,
                        nb_updates,
                        mode,
                    )
                    prev_state = current_state
                    nb_updates += 1

            for ID in to_cancel:
                self.ongoing_orders.pop(ID)

        print(f"Simulation runned for {t2 - t1:.2f}s", " " * 50)

        return (trades_list, market_event_list, self.actions_history, updates_list)

    def update_model(
        self,
        timestamp: float,
        reward: float,
        prev_state,
        prev_action,
        current_state,
        nb_updates: float,
        mode: str,
    ):

        if mode == "train":
            eps = self.epsilon_val(nb_updates)
        else:
            eps = 0.01

        current_action = self.model.choose_action(
            self.state_to_index(current_state), epsilon=eps
        )

        self.trajectory["rewards"].append((timestamp, reward))

        self.model.update(
            self.state_to_index(prev_state),
            prev_action,
            reward,
            self.state_to_index(current_state),
            current_action,
        )

        self.trajectory["eps"].append((timestamp, eps))
        self.trajectory["actions"].append(
            (timestamp, str(self.action_dict[current_action]))
        )

        return current_action

    def reset(self):

        self.inventory = self.initial_position
        self.realized_pnl = 0
        self.unrealized_pnl = 0

        self.actions_history = []
        self.ongoing_orders = {}

        self.trajectory = {
            "actions": [],
            "rewards": [],
            "eps": [],
        }

        self.model.initialize(
            [x[0] + 2 if x[3] else x[0] for x in self.state_space],
            len(self.action_dict),
        )

    def state_to_index(self, state_values):
        """
        Convert state space values to corresponding indices in the Q-table.

        Args:
            state_values (List[float]): List of state values
        """
        indices = []
        for i, val in enumerate(state_values):
            if self.state_space[i][3]:
                if self.state_space[i][3] and val < self.state_space[i][1]:
                    indices.append(0)
                    continue
                elif val > self.state_space[i][2]:
                    indices.append(self.state_space[i][0] + 1)
                    continue
            else:
                if val < self.state_space[i][1]:
                    val = self.state_space[i][1]
                elif val > self.state_space[i][2]:
                    val = self.state_space[i][2]

            # Normalize the state value to be between 0 and 1
            normalized_val = (val - self.state_space[i][1]) / (
                self.state_space[i][2] - self.state_space[i][1]
            )

            # Scale the normalized value to the number of levels and convert to an integer index
            index = int(
                normalized_val * (self.state_space[i][0] - 1) + self.state_space[i][3]
            )
            indices.append(index)

        return tuple(indices)

    def get_state(
        self, best_ask, best_bid, prices, asks_size, bids_size
    ) -> Tuple[float, float]:
        inv_ratio = inventory_ratio(
            self.inventory, self.min_position, self.max_position
        )
        book_imb = book_imbalance(asks_size, bids_size)
        spread = best_ask - best_bid
        vol = volatility(prices, 300)
        rsi = RSI(prices, 300)

        return (
            inv_ratio,
            book_imb,
            # spread,
            vol,
            rsi,
        )

    def epsilon_val(
        self,
        tick: float,
        alpha: float = 5,
        beta: float = 1,
        delta_reward: float = None,
    ):
        """
        Compute epsilon value for epsilon-greedy strategy.

        Args:
            tick (float): current tick
            alpha (float): slope of the logistic function
            beta (float): middle point of the logistic function
            delta_reward (float): reward difference
        """

        if delta_reward is None:
            Delta = 0
        else:
            Delta = abs(delta_reward) * 1e-5

        # Compute epsilon using logistic function
        return 1 / (1 + np.exp(alpha * (tick * 1e-5 - Delta) - beta)) + 0.1

    def get_expected_reward(self, state, action):
        return self.model.q_table[state + (action,)]

    def get_trajectory(self):
        df_trajectory = {}
        for key in self.trajectory.keys():
            if len(self.trajectory[key]) == 0:
                continue
            df_trajectory[key] = pd.DataFrame(
                np.array(self.trajectory[key]), columns=["timestamp", key]
            )

        return df_trajectory

    def get_q_table(self):
        return self.model.q_table

    def save_q_table(self, path):
        np.save(path, self.model.q_table)

    def load_q_table(self, path):
        self.model.q_table = np.load(path)
