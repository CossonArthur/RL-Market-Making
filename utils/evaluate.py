import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
import datetime
from typing import List, Union, Tuple

from environment.env import OwnTrade, MarketEvent, update_best_positions


def get_pnl(
    updates_list: List[Union[MarketEvent, OwnTrade]], maker_fee=0
) -> pd.DataFrame:
    """
    This function calculates PnL from list of updates
    """

    # current position in btc and usd
    inv, pnl = 0.0, 0.0

    N = len(updates_list)
    inv_arr = np.zeros((N,))
    pnl_arr = np.zeros((N,))
    mid_price_arr = np.zeros((N,))
    spread_arr = np.zeros((N,))

    # current best_bid and best_ask
    best_bid: float = -np.inf
    best_ask: float = np.inf

    for i, update in enumerate(updates_list):
        if isinstance(update, MarketEvent):
            best_bid, best_ask = update_best_positions(
                best_bid, best_ask, update, levels=False
            )

        if isinstance(update, OwnTrade):
            if update.execute == "TRADE":
                trade = update
                # update positions
                if trade.side == "buy":
                    inv += trade.size
                    pnl -= (1 + maker_fee) * trade.price * trade.size
                elif trade.side == "sell":
                    inv -= trade.size
                    pnl += (1 - maker_fee) * trade.price * trade.size

            # current portfolio value

        inv_arr[i] = inv
        pnl_arr[i] = pnl
        mid_price_arr[i] = (best_ask + best_bid) / 2
        spread_arr[i] = best_ask - best_bid

    worth_arr = inv_arr * mid_price_arr + pnl
    receive_ts = [update.receive_ts for update in updates_list]
    exchange_ts = [update.exchange_ts for update in updates_list]

    df = pd.DataFrame(
        {
            "exchange_ts": exchange_ts,
            "receive_ts": receive_ts,
            "total": worth_arr,
            "inventory": inv_arr,
            "PnL": pnl,
            "mid_price": mid_price_arr,
            "spread": spread_arr,
        }
    )
    df = df.groupby("receive_ts").agg(lambda x: x.iloc[-1]).reset_index()
    return df


def action_to_dataframe(action_list: List[Tuple]) -> pd.DataFrame:

    df = pd.DataFrame(
        action_list,
        columns=["receive_ts", "inventory", "action", "ask_price", "bid_price"],
    )

    df = df.groupby("receive_ts").agg(lambda x: x.iloc[-1]).reset_index()
    return df


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
    strategy,
    trades: List[OwnTrade],
    updates_list: List[Union[MarketEvent, OwnTrade]],
    orders: List[Tuple],
    trajectory: dict[List],
):

    trades_df = trade_to_dataframe(trades)
    orders_df = action_to_dataframe(orders)

    orders_df = orders_df.iloc[trades_df.shape[0] :]
    orders_df["spread"] = orders_df["ask_price"] - orders_df["bid_price"]
    orders_df["volatility"] = orders_df["spread"].rolling(window=100).std()
    orders_df = orders_df.dropna()

    pnl = get_pnl(updates_list, maker_fee=strategy.maker_fee)

    print(f"Mean PnL: ", pnl["total"].mean())

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, subplot_titles=("Price", "PnL", "Inventory")
    )

    fig.add_trace(
        go.Scatter(
            x=pnl["receive_ts"].apply(lambda x: datetime.datetime.fromtimestamp(x)),
            y=pnl["mid_price"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pnl["receive_ts"].apply(lambda x: datetime.datetime.fromtimestamp(x)),
            y=pnl["total"],
            name="PnL",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pnl["receive_ts"].apply(lambda x: datetime.datetime.fromtimestamp(x)),
            y=pnl["inventory"],
            name="Inventory",
        ),
        row=3,
        col=1,
    )
    fig.update_layout(title_text="Strategy Evaluation", template="plotly_dark")
    fig.show()

    # Inventory
    mean = pnl["inventory"].mean()
    std = pnl["inventory"].std()
    skew = pnl["inventory"].skew()
    print(f"Mean Inventory: {mean:.4f} - Std: {std:.4f} - Skew: {skew:.2f}")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Own Spread", "Spread", "Difference of spreads"),
    )
    fig.add_trace(
        go.Scatter(
            x=orders_df["receive_ts"].apply(
                lambda x: datetime.datetime.fromtimestamp(x)
            ),
            y=orders_df["spread"],
            name="Own spread",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pnl["receive_ts"].apply(lambda x: datetime.datetime.fromtimestamp(x)),
            y=pnl["spread"],
            name="Spread",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=orders_df["receive_ts"].apply(
                lambda x: datetime.datetime.fromtimestamp(x)
            ),
            y=orders_df["spread"] - pnl["spread"],
            name="Difference of spreads",
        ),
        row=3,
        col=1,
    )
    fig.update_layout(title_text="Spread", template="plotly_dark")
    fig.show()

    def action_parser(x):
        x = x.split(", ")
        return int(x[0][1:]), int(x[1][:-1])

    temp = []
    for action in trajectory["actions"]["actions"]:
        ask, bid = action_parser(action)
        temp.append((ask + 1, "ask"))
        temp.append((-bid - 1, "bid"))

    order_book = pd.DataFrame(temp, columns=["action", "side"])

    fig = px.histogram(
        order_book,
        x="action",
        color="side",
        template="plotly_dark",
        title="Number of actions",
    )
    fig.show()
