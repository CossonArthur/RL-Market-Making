import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import datetime
from typing import List

from environment.env import OwnTrade, MarketEvent, update_best_positions


def trade_to_dataframe(trades_list: List[OwnTrade]) -> pd.DataFrame:
    exchange_ts = [trade.exchange_ts for trade in trades_list]
    receive_ts = [trade.receive_ts for trade in trades_list]

    size = [trade.size for trade in trades_list]
    price = [trade.price for trade in trades_list]
    side = [trade.side for trade in trades_list]

    dct = {
        "exchange_ts": exchange_ts,
        "receive_ts": receive_ts,
        "size": size,
        "price": price,
        "side": side,
    }

    df = pd.DataFrame(dct).groupby("receive_ts").agg(lambda x: x.iloc[-1]).reset_index()
    return df


def md_to_dataframe(md_list: List[MarketEvent]) -> pd.DataFrame:
    best_bid = -np.inf
    best_ask = np.inf
    best_bids = []
    best_asks = []
    for md in md_list:
        best_bid, best_ask = update_best_positions(best_bid, best_ask, md)

        best_bids.append(best_bid)
        best_asks.append(best_ask)

    exchange_ts = [md.exchange_ts for md in md_list]
    receive_ts = [md.receive_ts for md in md_list]
    dct = {
        "exchange_ts": exchange_ts,
        "receive_ts": receive_ts,
        "bid_price": best_bids,
        "ask_price": best_asks,
    }

    df = pd.DataFrame(dct).groupby("receive_ts").agg(lambda x: x.iloc[-1]).reset_index()
    return df


# TODO: Add a function to evaluate the strategy graph etc
def evaluate_strategy(
    strategy, trades: pd.DataFrame, trajectory: dict, md_updates: pd.DataFrame
):

    print(f"Total PnL: {strategy.realized_pnl}")
    print(f"Number of Trades: {len(trades)}")

    fig = make_subplots(rows=3, cols=1, subplot_titles=("PnL", "Inventory"))

    fig.add_trace(
        go.Scatter(
            x=md_updates["receive_ts"].apply(
                lambda x: datetime.datetime.fromtimestamp(x)
            ),
            y=(md_updates["bid_price"] + md_updates["ask_price"]) / 2,
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(trajectory["realized_pnl"])),
            y=trajectory["realized_pnl"],
            name="PnL",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(trajectory["inventory"])),
            y=trajectory["inventory"],
            name="Inventory",
        ),
        row=3,
        col=1,
    )
    fig.show()
