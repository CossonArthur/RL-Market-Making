import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
from statsmodels.tsa.stattools import adfuller
from typing import List, Union, Tuple

from environment.env import OwnTrade, MarketEvent, update_best_positions


def get_pnl(updates_list: List[Union[MarketEvent, OwnTrade]]) -> pd.DataFrame:
    """
    This function calculates PnL from list of updates
    """

    # current position in btc and usd
    inv = 0.0

    N = len(updates_list)
    inv_arr = np.zeros((N,))
    pnl_arr = np.zeros((N,))
    mid_price_arr = np.zeros((N,))
    spread_arr = np.zeros((N,))

    # current best_bid and best_ask
    best_bid: float = -np.inf
    best_ask: float = np.inf

    for i, update in enumerate(updates_list):

        spread_arr[i] = best_ask - best_bid
        mid_price_arr[i] = (best_ask + best_bid) / 2

        if isinstance(update, MarketEvent):
            best_bid, best_ask = update_best_positions(
                best_bid, best_ask, update, levels=False
            )

        if isinstance(update, OwnTrade):
            if update.execute == "TRADE":

                if update.side == "buy":
                    inv += update.size
                    pnl_arr[i] = (mid_price_arr[i] - update.price) * update.size

                elif update.side == "sell":
                    inv -= update.size
                    pnl_arr[i] = (update.price - mid_price_arr[i]) * update.size

        inv_arr[i] = inv

    receive_ts = [update.receive_ts for update in updates_list]
    exchange_ts = [update.exchange_ts for update in updates_list]

    df = pd.DataFrame(
        {
            "exchange_ts": exchange_ts,
            "receive_ts": receive_ts,
            "inventory": inv_arr,
            "PnL": pnl_arr,
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


def adf_test(timeseries):

    dftest = adfuller(timeseries)
    return ("stationnary" if dftest[1] < 0.05 else "non-stationnary", dftest[1])


def evaluate_strategy(
    strategy,
    trades: List[OwnTrade],
    updates_list: List[Union[MarketEvent, OwnTrade]],
    orders: List[Tuple],
):

    pnl = get_pnl(updates_list)
    pnl["receive_ts"] = pnl["receive_ts"].apply(
        lambda x: datetime.datetime.fromtimestamp(x)
    )
    pnl = pnl.loc[pnl["PnL"] >= 0]

    print(f"Executed Trades: {len([x for x in trades if x.execute == 'TRADE']):.0f}")
    print(f"Mean PnL: {pnl['PnL'].mean():.4f}")

    fig, axs = plt.subplots(3, 2, figsize=(15, 10), width_ratios=[2, 1])

    axs[0, 0].plot(pnl["receive_ts"], pnl["mid_price"], label="Price")
    axs[0, 0].set_title("Price")
    axs[0, 0].grid()
    axs[0, 0].legend(loc="lower left")

    mid_return = pnl["mid_price"].diff().dropna() / pnl["mid_price"].shift(1).dropna()
    # remove teh outliers
    mid_return = mid_return[(mid_return > -0.00005) & (mid_return < 0.00005)]

    # Plot the hist of the corr_return
    axs[0, 1].hist(mid_return, bins=100)
    axs[0, 1].set_title(
        f"mean: {mid_return.mean():.4f}, std: {mid_return.std():.4f}, skew: {mid_return.skew():.2f}"
    )
    axs[0, 1].set_xlabel("return")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].grid()

    axs[1, 0].plot(pnl["receive_ts"], pnl["PnL"], label="PnL", color="orange")
    axs[1, 0].set_title("PnL")
    axs[1, 0].grid()
    axs[1, 0].legend(loc="lower left")

    axs[1, 1].hist(pnl["PnL"], bins=100)
    axs[1, 1].set_title(
        f"mean: {pnl['PnL'].mean():.4f}, std: {pnl['PnL'].std():.4f}, skew: {pnl['PnL'].skew():.2f}"
    )
    axs[1, 1].set_xlabel("PnL")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].grid()

    axs[2, 0].plot(
        pnl["receive_ts"], pnl["inventory"], label="Inventory", color="green"
    )
    axs[2, 0].axhline(y=strategy.min_position, color="red", linestyle="--")
    axs[2, 0].axhline(y=strategy.max_position, color="red", linestyle="--")
    axs[2, 0].set_title("Inventory")
    axs[2, 0].grid()
    axs[2, 0].legend(loc="upper left")

    axs[2, 1].hist(pnl["inventory"], bins=100)
    axs[2, 1].axvline(x=strategy.min_position, color="red", linestyle="--")
    axs[2, 1].axvline(x=strategy.max_position, color="red", linestyle="--")
    axs[2, 1].set_title(
        f"mean: {pnl['inventory'].mean():.2f}, std: {pnl['inventory'].std():.2f}, skew: {pnl['inventory'].skew():.2f}"
    )
    axs[2, 1].set_xlabel("Inventory")
    axs[2, 1].set_ylabel("Frequency")
    axs[2, 1].grid()

    fig.suptitle("Strategy Evaluation", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    # Inventory ADF Test
    result, p_value = adf_test(
        pnl.loc[
            3 * len(pnl["inventory"]) // 8 : len(pnl["inventory"]) // 2, "inventory"
        ]
    )

    print(f"ADF Test for inventory serie : {result} - p-value: {p_value:.4f}")

    def action_parser(x):
        x = x.split(", ")
        return int(x[0][1:]), int(x[1][:-1])

    try:
        trajectory = strategy.get_trajectory()

        temp = []
        for action in trajectory["actions"]["actions"]:
            ask, bid = action_parser(action)
            temp.append((ask + 1, "ask"))
            temp.append((-bid - 1, "bid"))

        order_book = pd.DataFrame(temp, columns=["action", "side"])

        plt.figure(figsize=(10, 6))
        sns.histplot(data=order_book, x="action", hue="side", multiple="stack")

        plt.title("Order executed by the strategy", fontsize=16)
        plt.grid()
        plt.show()
    except AttributeError:
        return
