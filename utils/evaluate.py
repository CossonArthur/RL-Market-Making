import plotly.express as px

from environment.env import Sim
from strategies.rl import RLStrategy


# TODO: Add a function to evaluate the strategy graph etc
def evaluate_strategy(strategy: RLStrategy, sim: Sim, mode: str, count: int = 10000):
    trades, md_updates, orders, trajectory = strategy.run(sim, mode, count)

    print(f"Total PnL: {strategy.realized_pnl}")
    print(f"Number of Trades: {len(trades)}")

    fig = px.line(
        trajectory["rewards"], x="tick", y="Rewards", title="Rewards over time"
    )
    fig.show()
