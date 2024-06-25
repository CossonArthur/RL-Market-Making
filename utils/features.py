import pandas as pd
import numpy as np

from typing import List


def inventory_ratio(inventory: float, min_inventory: float, max_inventory: float):
    if max_inventory - min_inventory == 0:
        return (inventory - min_inventory) / 1e-12

    return (inventory - min_inventory) / (max_inventory - min_inventory)


def volatility(price: List, n: int = 1000) -> float:
    """
    Calculate the volatility of a price series.

    Args:
    price (List): Price series.
    n (int): Number of periods for the moving average.
    """

    if len(price) < n + 1:
        return 0

    return np.std(np.diff(price[-n:]) / price[-n:-1])


def RSI(price: List, n: int = 300) -> float:
    """
    Calculate the Relative Strength Index (RSI) of a price series. using the EMAs of the gains and losses.

    Args:
    price (List): Price series.
    time_delta (str): Time delta for resampling.
    """

    if len(price) < n + 1:
        n = len(price) - 1

    def EMA(data, window):
        alpha = 2 / (window + 1.0)

        for i in range(1, len(data)):
            data[i] = alpha * data[i] + (1 - alpha) * data[i - 1]

        return data[-1]

    delta = np.diff(price[-n - 1 :])
    gain = delta[delta > 0]
    loss = -delta[delta < 0]

    avg_gain = EMA(gain, n)
    avg_loss = abs(EMA(loss, n))

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

    return (bids_size - asks_size) / (asks_size + bids_size)
