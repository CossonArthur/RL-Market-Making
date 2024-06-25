from environment.env import Real_Data_Env
from strategies.rl import QLearning, RLStrategy, SARSA
from strategies.baselines import BestPosStrategy, StoikovStrategy
from utils.load_data import load_data
from utils.evaluate import evaluate_strategy

import numpy as np
import pandas as pd
import plotly.express as px
import datetime

market_data = load_data()

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
plt

books["mid_price"] = (books["bid_0_price"] + books["ask_0_price"]) / 2
books["received_time"] = pd.to_datetime(books["received_time"], unit="s")

sns.lineplot(
    data=books[500:700],
    x="received_time",
    y="mid_price",
    color="blue",
    label="Middle price",
)
sns.lineplot(
    data=books[500:700],
    x="received_time",
    y="ask_0_price",
    color="red",
    label="Best ask",
)
sns.lineplot(
    data=books[500:700],
    x="received_time",
    y="bid_0_price",
    color="green",
    label="Best bid",
)

plt.legend()
plt.xlabel("Time")
plt.ylabel("USDT")

# plot the order book with a bar chart from the depth 0 to 20 at index 700
plt.bar(
    x=books.filter(regex="ask_[0-9]+_price").iloc[700],
    height=books.filter(regex="ask_[0-9]+_size").iloc[700],
    color="red",
    alpha=0.7,
    label="Ask",
    width=0.1,
)

# line for the mid price
plt.axvline(
    x=books["mid_price"].iloc[700],
    color="blue",
    linestyle="--",
    label="Mid price",
)

plt.bar(
    x=books.filter(regex="bid_[0-9]+_price").iloc[700],
    height=books.filter(regex="bid_[0-9]+_size").iloc[700],
    color="green",
    alpha=0.7,
    label="Bid",
    width=0.1,
)

plt.xlabel("Price Level")
plt.ylabel("Size")
plt.legend()
plt.show()


# Configuration and initialization
q_learning = QLearning()
sarsa = SARSA()


min_position = -1  # Example: minimum position size
max_position = 1  # Example: maximum position size
delay = 5e-2
trade_size = 0.001
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

# strategy = StoikovStrategy(
#     min_position=min_position,
#     max_position=max_position,
#     delay=delay,
#     trade_size=trade_size,
#     maker_fee=maker_fee,
# )

# Create the env
sim = Real_Data_Env(market_data, 1e-4, 1e-4)


# # Train and evaluate the strategy
trades, market_updates, orders, updates = strategy.run(sim, "train", 500000)
# evaluate_strategy(strategy, trades, updates, orders)

# # # # Save Q-table
# time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# strategy.save_q_table(f"model/{str(strategy.model)}_{time}.npy")

# strategy.reset()

# # # Load Q-table for evaluation or further training
# strategy.load_q_table(f"model/{str(strategy.model)}_{time}.npy")


# # Evaluate in test mode
# trades, market_updates, orders, updates = strategy.run(sim, 1000000)
# evaluate_strategy(strategy, trades, updates, orders)
