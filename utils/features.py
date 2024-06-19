import pandas as pd
import numpy as np

from typing import List


def volatility(price: List, n: int = 300, m: int = 20) -> float:
    """
    Calculate the volatility of a price series.
    Using the formulat of Patel, Y. "Optimizing market making using multi-agent reinforcement learning"

    Args:
    price (List): Price series.
    n (int): Relative starting point for the EMA from the end
    m (int): Relative ending point for the EMA from the end
    """

    def EMA(price: pd.DataFrame, n: int = 300, m: int = 0) -> float:
        """
        Calculate the Exponential Moving Average (EMA) of a price serie as in Patel, Y. "Optimizing market making using multi-agent reinforcement learning"
        """

        assert n > m, "n should be less than m"

        if len(price) < n + 1 or len(price) < m + 1:
            return 1e-9

        return 2 * price[-1 - m] / (n + 1) + sum(price[-n - 1 : -1 - m]) / n * (
            100 - 2 / (n + 1)
        )

    return (EMA(price, n, 0) - EMA(price, n, m)) / EMA(price, n, m)


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

        pows = alpha_rev ** (np.arange(n + 1))

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

    avg_gain = EMA(gain, n)
    avg_loss = EMA(loss, n)

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
        return 0.0

    return (asks_size - bids_size) / (asks_size + bids_size)
