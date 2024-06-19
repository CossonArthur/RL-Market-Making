import numpy as np
import pandas as pd
from environment.env import Sim
from strategies.rl import QLearning, RLStrategy, evaluate_strategy
from utils.load_data import load_data

market_data = load_data()

# Configuration and initialization
q_learning = QLearning()


min_position = 0  # Example: minimum position size
max_position = 1000  # Example: maximum position size
delay = 5e-6
trade_size = 0.001
maker_fee = -0.00004

# Initialize strategy
strategy = RLStrategy(
    model=q_learning,
    min_position=min_position,
    max_position=max_position,
    delay=delay,
    trade_size=trade_size,
    maker_fee=maker_fee,
    order_book_depth=5,
)

# Create a simulator instance (assuming Sim class is properly defined in simulator module)
sim = Sim(market_data, "train", 10)

# Train and evaluate the strategy
mode = "train"
total_pnl, num_trades = evaluate_strategy(strategy, sim, mode)

print("Training done!")
print(f"Total PnL: {total_pnl}")
print(f"Number of Trades: {num_trades}")

# # Save Q-table
# strategy.save_q_table("q_table.npy")

# # Load Q-table for evaluation or further training
# strategy.load_q_table("q_table.npy")

# # Evaluate in test mode
# mode = "test"
# total_pnl, num_trades = evaluate_strategy(strategy, sim, mode)
