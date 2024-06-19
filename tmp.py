import numpy as np
import pandas as pd
from environment.env import Sim
from strategies.rl import QLearning, RLStrategy, evaluate_strategy
from utils.load_data import load_data


market_data = load_data()

# Configuration and initialization
n_actions = 10  # Example: number of discrete actions
n_states = 300  # Example: number of discrete states
q_learning = QLearning(n_actions, n_states)

# Example DataFrame for features
ess_df = pd.DataFrame(
    {
        "receive_ts": pd.date_range(start="1/1/2023", periods=300, freq="S"),
        "feature1": np.random.randn(300),
        "feature2": np.random.randn(300),
        # Add more features as needed
    }
)

min_position = 0  # Example: minimum position size
max_position = 1000  # Example: maximum position size
delay = 10e9  # 1 second in nanoseconds
hold_time = 5 * 10e9  # 5 seconds in nanoseconds
trade_size = 0.001  # Example trade size
taker_fee = 0.0004  # Example taker fee
maker_fee = -0.00004  # Example maker fee

# Initialize strategy
strategy = RLStrategy(
    model=q_learning,
    ess_df=ess_df,
    min_position=min_position,
    max_position=max_position,
    delay=delay,
    hold_time=hold_time,
    trade_size=trade_size,
    taker_fee=taker_fee,
    maker_fee=maker_fee,
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
