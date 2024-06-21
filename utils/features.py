import pandas as pd
import numpy as np

from typing import List


def inventory_ratio(inventory: float, min_inventory: float, max_inventory: float):
    if max_inventory - min_inventory == 0:
        return (inventory - min_inventory) / 1e-12

    return (inventory - min_inventory) / (max_inventory - min_inventory)


def volatility(price: List, n: int = 300) -> float:
    """
    Calculate the volatility of a price series.

    Args:
    price (List): Price series.
    n (int): Number of periods for the moving average.
    """
    # TODO : fix this

    if len(price) < n + 1:
        return 0

    return np.std(np.diff(price[-n:]))


def RSI(price: List, n: int = 300) -> float:
    """
    Calculate the Relative Strength Index (RSI) of a price series. using the EMAs of the gains and losses.

    Args:
    price (List): Price series.
    time_delta (str): Time delta for resampling.
    """

    if len(price) < n + 1:
        return 50

    def EMA(data, window):
        alpha = 2 / (window + 1.0)
        alpha_rev = 1 - alpha
        n = data.shape[0]

        pows = (1 - alpha) ** (np.arange(n + 1))

        scale_arr = 1 / pows[:-1]
        offset = data[0] * pows[1:]
        pw0 = alpha * alpha_rev ** (n - 1)

        mult = data * pw0 * scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums * scale_arr[::-1]
        return out

    delta = np.diff(price[-n - 1 :])
    gain = delta[delta > 0]
    loss = -delta[delta < 0]

    avg_gain = EMA(gain, n)[-1]
    avg_loss = EMA(loss, n)[-1]

    rs = avg_gain / avg_loss

    return 100 - 100 / (1 + rs)


def book_imbalance(asks_volume, bids_volume) -> float:
    """
    Calculate the book imbalance.

    Args:
    asks_volume (List): List of ask prices and sizes.
    bids_volume (List): List of bid prices and sizes.
    """

    bids_size = sum(bids_volume)
    asks_size = sum(asks_volume)

    if asks_size + bids_size == 0:
        return 0

    return (asks_size - bids_size) / (asks_size + bids_size)
