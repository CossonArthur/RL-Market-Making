from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import datetime

from environment.env import OwnTrade, Sim, MarketEvent, Order, update_best_positions
from utils.features import book_imbalance, RSI, volatility


class QLearning:
    def __init__(self, alpha=0.1, gamma=0.99):
        self.alpha = alpha
        self.gamma = gamma

    def initialize(self, state_sizes, n_actions):
        self.q_table = np.zeros(state_sizes + [n_actions])

    def choose_action(self, state, epsilon=0.1):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state, :])

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
            return np.random.choice(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state, :])

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
        hold_time: Optional[float] = None,
        trade_size: float = 0.001,
        maker_fee: float = -0.00004,
        order_book_depth: int = 10,
    ) -> None:
        """
        Args:
            model(Union[QLearning]): RL model
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
            hold_time = min(delay * 5, pd.Timedelta(10, "s").delta)
        self.hold_time = hold_time

        self.coin_position = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.actions_history = []
        self.ongoing_orders = {}

        self.action_dict = {  # id : (ask_level, bid_level)
            i + j + 1: (i, j)
            for i in range(order_book_depth + 1)
            for j in range(order_book_depth + 1)
        }

        self.state_space = [  # level, min, max for each feature
            (10, 0, 1),  # inventory ratio
            (10, -1, 1),  # book imbalance
            (10, 0, 1),  # spread #  TODO: Define relevant state space
            (10, 0, 1),  # volatility #  TODO: Define relevant state space
            (10, 0, 100),  # rsi
        ]

        self.trajectory = {
            key: []
            for key in [
                "actions",
                "observations",
                "rewards",
            ]
        }

    def place_order(
        self, sim: Sim, action_id: float, receive_ts: float, asks_price, bids_price
    ):
        if action_id == 0:
            return

        else:
            ask_level, bid_level = self.action_dict[action_id]
            ask_order = sim.place_order(
                receive_ts, self.trade_size, "sell", asks_price[ask_level]
            )
            bid_order = sim.place_order(
                receive_ts, self.trade_size, "buy", bids_price[bid_level]
            )

            self.ongoing_orders[bid_order.order_id] = (bid_order, "LIMIT")
            self.ongoing_orders[ask_order.order_id] = (ask_order, "LIMIT")

        self.actions_history.append((receive_ts, self.coin_position, action_id))

    def run(self, sim: Sim, mode: str, count=10) -> Tuple[
        List[OwnTrade],
        List[MarketEvent],
        List[Union[OwnTrade, MarketEvent]],
        List[Order],
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
            [x[0] + 2 for x in self.state_space],
            len(self.action_dict) + 1,  # level + 2 for val beyond max and min
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
        prev_total_pnl = None

        current_state = self.get_state(best_ask, best_bid, [], [], [])
        prev_state = current_state
        current_action = None
        prev_action = None

        if mode != "train":
            count = 1e8

        t1 = datetime.datetime.now().timestamp()

        while len(self.trajectory["rewards"]) < count:
            # get update from simulator
            t2 = datetime.datetime.now().timestamp()
            receive_ts, updates = sim.tick()
            print(
                f"Elapsed time: {t2 - t1:.2f}s \n",
                f"Time: {receive_ts/1e9:.2f}s \n",
                f"Number of rewards: {len(self.trajectory['rewards'])}",
                end="\r",
            )

            if updates is None:
                break

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
                        if update.side == "BID":
                            self.coin_position += update.size
                            self.realized_pnl -= (
                                (1 + self.maker_fee) * update.price * update.size
                            )
                        else:
                            self.coin_position -= update.size
                            self.realized_pnl += (
                                (1 - self.maker_fee) * update.price * update.size
                            )
                        self.unrealized_pnl = self.coin_position * (
                            (best_ask + best_bid) / 2
                        )

                else:
                    assert False, "invalid type of update!"

            # if the delay has passed, place an order
            if receive_ts - prev_time >= self.delay:

                # update state
                prev_state = current_state
                current_state = self.get_state(
                    best_ask, best_bid, sim.price_history, asks_volume, bids_volume
                )
                self.trajectory["observations"].append(current_state)

                # choose action
                prev_action = current_action
                current_action = self.model.choose_action(
                    self.state_to_index(current_state)
                )

                self.trajectory["actions"].append(
                    self.action_dict[current_action] if current_action != 0 else None
                )

                self.place_order(
                    sim, current_action, receive_ts, asks_price, bids_price
                )

                if mode == "train":
                    if prev_total_pnl is None:
                        prev_total_pnl = 0
                        prev_coin_pos = 0
                    else:
                        prev_total_pnl = self.realized_pnl + self.unrealized_pnl
                        prev_coin_pos = self.coin_position

                    # TODO: calculate reward
                    reward = (
                        self.realized_pnl
                        + self.unrealized_pnl
                        - prev_total_pnl
                        - abs(prev_coin_pos)
                    )
                    self.trajectory["rewards"].append(reward)

                    self.model.update(
                        self.state_to_index(prev_state),
                        prev_action,
                        reward,
                        self.state_to_index(current_state),
                        current_action,
                    )

                prev_time = receive_ts

            to_cancel = []
            for ID, (order, order_type) in self.ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                self.ongoing_orders.pop(ID)

        return trades_list, md_list, updates_list, self.actions_history, self.trajectory

    def reset(self):

        self.coin_position = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0

        self.actions_history = []
        self.ongoing_orders = {}

        self.trajectory = {
            key: []
            for key in [
                "actions",
                "observations",
                "rewards",
            ]
        }

    def state_to_index(self, state_values):
        """
        Convert state space values to corresponding indices in the Q-table.

        Args:
            state_values (List[float]): List of state values
        """
        indices = []
        for i, val in enumerate(state_values):
            if val < self.state_space[i][1]:
                indices.append(0)
                continue
            elif val > self.state_space[i][2]:
                indices.append(self.state_space[i][0])
                continue

            # Normalize the state value to be between 0 and 1
            normalized_val = (val - self.state_space[i][1]) / (
                self.state_space[i][2] - self.state_space[i][1]
            )

            # Scale the normalized value to the number of levels and convert to an integer index
            index = int(normalized_val * (self.state_space[i][0] - 1) + 1)
            indices.append(index)

        return tuple(indices)

    def get_state(
        self, best_ask, best_bid, prices, asks_size, bids_size
    ) -> Tuple[float, float]:
        inventory_ratio = self.coin_position - self.min_position / (
            self.max_position - self.min_position
        )

        spread = best_ask - best_bid

        vol = volatility(prices, 300)

        rsi = RSI(prices, 300)

        book_imb = book_imbalance(asks_size, bids_size)

        return (
            inventory_ratio,
            spread,
            vol,
            rsi,
            book_imb,
        )

    def get_expected_reward(self, state, action):
        return self.model.q_table[state, action]

    def get_q_table(self):
        return self.model.q_table

    def save_q_table(self, path):
        np.save(path, self.model.q_table)

    def load_q_table(self, path):
        self.model.q_table = np.load(path)


# TODO: Add a function to evaluate the strategy graph etc
def evaluate_strategy(strategy, sim, mode, count=1000):
    trades, md_updates, all_updates, orders = strategy.run(sim, mode, count)

    total_pnl = strategy.realized_pnl + strategy.unrealized_pnl
    num_trades = len(trades)

    print(f"Total PnL: {total_pnl}")
    print(f"Number of Trades: {num_trades}")

    return total_pnl, num_trades


# # Define parameters
# alpha = 1.0  # Steepness of logistic function
# beta = 0.5  # Midpoint of logistic function


# def epsilon_var(alpha: float = 1.0, beta: float = 0.5):
#     r_t = get_actual_reward()
#     hat_r_t = get_expected_reward()
#     Delta_t = r_t - hat_r_t

#     # Compute epsilon using logistic function
#     return 1 / (1 + np.exp(-alpha * (Delta_t - beta)))
