from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import datetime

from environment.env import OwnTrade, Sim, MarketEvent, Order, update_best_positions
from utils.features import book_imbalance, RSI, volatility
from utils.evaluate import trade_to_dataframe, md_to_dataframe


class QLearning:
    def __init__(self, alpha=0.1, gamma=0.99):
        self.alpha = alpha
        self.gamma = gamma

    def initialize(self, state_sizes, n_actions):
        self.q_table = np.zeros(state_sizes + [n_actions])

    def choose_action(self, state, epsilon=0.1):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.q_table.shape[-1])
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action):
        self.q_table[state + (action,)] += self.alpha * (
            reward
            + self.gamma
            * self.q_table[next_state + (np.argmax(self.q_table[next_state]),)]
            - self.q_table[state + (action,)]
        )


class SARSA:
    def __init__(self, alpha=0.1, gamma=0.99):
        self.alpha = alpha
        self.gamma = gamma

    def initialize(self, state_sizes, n_actions):
        self.q_table = np.zeros(state_sizes + [n_actions])

    def choose_action(self, state, epsilon=0.1):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.q_table.shape[-1])
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action):
        self.q_table[state + (action,)] += self.alpha * (
            reward
            + self.gamma * self.q_table[next_state + (next_action,)]
            - self.q_table[state + (action,)]
        )


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
        trade_size: float = 0.001,
        maker_fee: float = -0.00004,
        order_book_depth: int = 10,
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
        self.model = model

        self.maker_fee = maker_fee
        self.trade_size = trade_size
        self.min_position = min_position
        self.max_position = max_position
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, 5e-2)
        self.hold_time = hold_time

        if initial_position is None:
            initial_position = (max_position + min_position) / 2
        self.inventory = initial_position
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.actions_history = []
        self.ongoing_orders = {}

        self.action_dict = {  # id : (ask_level, bid_level)
            i: x
            for (i, x) in enumerate(
                [
                    (i, j)
                    for i in range(order_book_depth + 1)
                    for j in range(order_book_depth + 1)
                ]
            )
        }

        self.state_space = [  # level, min, max for each feature, bin for extreme values
            (3, 0, 1, True),  # inventory ratio
            # (5, -1, 1, False),  # book imbalance (mostly 0 here)
            # (10, 0, 1, False),  # spread #  TODO: Define relevant state space
            # (10, 0, 1, False),  # volatility #  TODO: Define relevant state space
            # (10, 0, 100, False),  # rsi
        ]

        self.trajectory = {
            "actions": [],
            "observations": [],
            "rewards": [],
            "realized_pnl": [],
            "inventory": [],
        }

    def place_order(
        self, sim: Sim, action_id: float, receive_ts: float, asks_price, bids_price
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
                datetime.datetime.fromtimestamp(receive_ts),
                self.inventory,
                self.action_dict[action_id],
            )
        )

    def run(self, sim: Sim, mode: str, count=10) -> Tuple[
        List[OwnTrade],
        List[MarketEvent],
        List[Order],
        dict,
    ]:
        """
        This function runs simulation

        Args:
            sim(Sim): simulator
        Returns:
            trades_list(List[OwnTrade]): list of our executed trades
            md_list(List[MarketEvent]): list of market data received by strategy
            updates_list( List[ Union[OwnTrade, MarketEvent] ] ): list of all updates
            received by strategy(market data and information about executed trades)
            all_orders(List[Order]): list of all placed orders
        """

        self.model.initialize(
            [x[0] + 2 if x[3] else x[0] for x in self.state_space],
            len(self.action_dict),
        )

        md_list: List[MarketEvent] = []
        trades_list: List[OwnTrade] = []
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
        prev_total_pnl = 0

        current_state = self.get_state(best_ask, best_bid, [], [], [])
        prev_state = current_state
        current_action = None
        prev_action = None

        if mode != "train":
            count = 1e8

        t1 = datetime.datetime.now().timestamp()
        t2 = t1

        tick = 0
        while tick < count:
            # get update from simulator
            t2 = datetime.datetime.now().timestamp()
            receive_ts, updates = sim.tick()
            tick += 1

            if updates is None:
                break

            simulated_time = (
                receive_ts
                - datetime.datetime(year=2022, month=10, day=1, hour=2).timestamp()
            )
            print(
                f"Elapsed time: {t2 - t1:.2f}s",
                f"Simulated time: {simulated_time//3600:.2f}h {(simulated_time%3600)//60:.2f}m {simulated_time%60:.2f}s",
                " " * 10,
                end="\r",
            )
            # save updates
            updates_list += updates
            for update in updates:

                # if update is market data, update best position
                if isinstance(update, MarketEvent):
                    if update.orderbook is not None:
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
                    md_list.append(update)

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
                            self.realized_pnl += (
                                (1 - self.maker_fee) * update.price * update.size
                            )
                        self.unrealized_pnl = self.inventory * (
                            (best_ask + best_bid) / 2
                        )

                    self.trajectory["realized_pnl"].append(
                        (receive_ts, self.realized_pnl)
                    )
                    self.trajectory["inventory"].append((receive_ts, self.inventory))

                    if mode == "train":
                        reward = (
                            self.realized_pnl + self.unrealized_pnl - prev_total_pnl
                        )

                        # penalize the agent for having a position too close to the limits (mean-reverting strategy)
                        # reward += -1e5 / (
                        #     1
                        #     + np.exp(
                        #         -20
                        #         * abs(
                        #             (self.inventory - self.min_position)
                        #             / (self.max_position - self.min_position)
                        #             - 0.5
                        #         )
                        #         + 5
                        #     )
                        # )

                        prev_total_pnl = self.realized_pnl + self.unrealized_pnl
                        self.trajectory["rewards"].append((receive_ts, reward))

                        self.model.update(
                            self.state_to_index(prev_state),
                            prev_action,
                            reward,
                            self.state_to_index(current_state),
                            current_action,
                        )

                else:
                    assert False, "invalid type of update!"

            # if the delay has passed, place an order
            if receive_ts - prev_time >= self.delay and len(self.ongoing_orders) == 0:

                # update state
                prev_state = current_state
                current_state = self.get_state(
                    best_ask, best_bid, sim.price_history, asks_volume, bids_volume
                )

                self.trajectory["observations"].append((receive_ts, current_state))
                # choose action
                prev_action = current_action
                current_action = self.model.choose_action(
                    self.state_to_index(current_state)
                )

                self.trajectory["actions"].append((receive_ts, current_action))

                self.place_order(
                    sim, current_action, receive_ts, asks_price, bids_price
                )

                prev_time = receive_ts

            to_cancel = []
            for ID, (order, order_type) in self.ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                self.ongoing_orders.pop(ID)

        print(f"Simulation runned for {t2 - t1:.2f}s", " " * 50)

        for key in self.trajectory.keys():
            self.trajectory[key] = pd.DataFrame(
                np.array(self.trajectory[key]), columns=["timestamp", key]
            )

        return (
            trade_to_dataframe(trades_list),
            md_to_dataframe(md_list),
            self.actions_history,
            self.trajectory,
        )

    def reset(self):

        if initial_position is None:
            initial_position = (self.max_position + self.min_position) / 2
        self.inventory = initial_position
        self.realized_pnl = 0
        self.unrealized_pnl = 0

        self.actions_history = []
        self.ongoing_orders = {}

        self.trajectory = {
            "actions": [],
            "observations": [],
            "rewards": [],
            "realized_pnl": [],
            "inventory": [],
        }

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
        inventory_ratio = (self.inventory - self.min_position) / (
            self.max_position - self.min_position
        )
        book_imb = book_imbalance(asks_size, bids_size)
        spread = best_ask - best_bid
        vol = volatility(prices, 300)
        rsi = RSI(prices, 300)

        return (
            inventory_ratio,
            # book_imb,
            # spread,
            # vol,
            # rsi,
        )

    def epsilon_var(self, alpha: float = 1.0, beta: float = 0.5):
        # TODO: Implement the epsilon variation function
        r_t = get_actual_reward()
        hat_r_t = get_expected_reward()
        Delta_t = r_t - hat_r_t

        # Compute epsilon using logistic function
        return 1 / (1 + np.exp(-alpha * (Delta_t - beta)))

    def get_expected_reward(self, state, action):
        return self.model.q_table[state + (action,)]

    def get_q_table(self):
        return self.model.q_table

    def save_q_table(self, path):
        np.save(path, self.model.q_table)

    def load_q_table(self, path):
        self.model.q_table = np.load(path)
