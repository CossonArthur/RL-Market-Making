from environment.env import Real_Data_Env
from strategies.rl import QLearning, RLStrategy, SARSA
from utils.load_data import load_data
from utils.evaluate import evaluate_strategy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

market_data = load_data()

# Configuration and initialization
q_learning = QLearning()
sarsa = SARSA()


min_position = -0.1  # Example: minimum position size
max_position = 0.1  # Example: maximum position size
delay = 5e-2
trade_size = 0.01
maker_fee = 0  # -0.00004

# Initialize strategy
strategy = RLStrategy(
    model=q_learning,
    min_position=min_position,
    max_position=max_position,
    delay=delay,
    trade_size=trade_size,
    maker_fee=maker_fee,
    order_book_depth=4,
)

# Create the env
sim = Real_Data_Env(market_data, 1e-4, 1e-4)


# Train and evaluate the strategy
trades, market_updates, orders, updates, trajectory = strategy.run(sim, "train", 100000)
evaluate_strategy(strategy, trades, updates, orders, trajectory)

# # # Save Q-table
# time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# strategy.save_q_table(f"model/{str(strategy.model)}_{time}.npy")

# # # # Load Q-table for evaluation or further training
# strategy.load_q_table(f"model/{str(strategy.model)}_{time}.npy")


# # Evaluate in test mode
# trades, market_updates, orders, updates, trajectory = strategy.run(sim, "test", 1000000)

# evaluate_strategy(strategy, trades, updates, orders, trajectory)
