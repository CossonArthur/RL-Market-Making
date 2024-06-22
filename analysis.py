import lakeapi
import datetime
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.api import OLS
from utils.features import inventory_ratio, volatility, RSI, book_imbalance

lakeapi.use_sample_data(anonymous_access=True)

# Load data
books = lakeapi.load_data(
    table="book",
    start=datetime.datetime(2022, 10, 1),
    end=datetime.datetime(2022, 10, 2),
    symbols=["BTC-USDT"],
    exchanges=["BINANCE"],
)

books["mid_price"] = (books["ask_0_price"] + books["bid_0_price"]) / 2
books["target_100"] = books["mid_price"].diff(100) / books["mid_price"]
books["target_1000"] = books["mid_price"].diff(1000) / books["mid_price"]
books["target_10000"] = books["mid_price"].diff(10000) / books["mid_price"]

#  split data in train and test
books["set"] = ["train"] * int(0.8 * len(books)) + ["test"] * int(0.2 * len(books))


def r2_out_of_sample(y_col: str, X_col: list[str]):
    data = books.dropna(
        subset=X_col + ["target_100", "target_1000", "target_10000", "set"]
    )

    if len(X_col) == 1:
        X_train = data.loc[data["set"] == "train", X_col].values.reshape(-1, 1)
        X_test = data.loc[data["set"] == "test", X_col].values.reshape(-1, 1)

    else:
        X_train = data.loc[data["set"] == "train", X_col].values
        X_test = data.loc[data["set"] == "test", X_col].values

    y_train = data.loc[data["set"] == "train", y_col].values
    y_test = data.loc[data["set"] == "test", y_col].values

    model = OLS(y_train, X_train)
    results = model.fit()

    predictions = results.predict(X_test)

    return 1 - np.sum((y_test - predictions) ** 2) / np.sum(
        (y_test - np.mean(y_test)) ** 2
    )


# Book Imbalance

asks_size = books.filter(regex="ask_[0-9]+_size")
bids_size = books.filter(regex="bid_[0-9]+_size")

bids_size = np.sum(bids_size, axis=1)
asks_size = np.sum(asks_size, axis=1)

books["imb"] = (asks_size - bids_size) / (asks_size + bids_size)


books["imb"].rolling(30000).corr(books["target_100"]).dropna().plot()


R2 = []

for target in ["target_100", "target_1000", "target_10000"]:
    R2.append(r2_out_of_sample(target, ["imb"]))

plt.scatter(["target_100", "target_1000", "target_10000"], R2)
plt.title("R2 for different targets vs imb")
plt.show()


# spread
books["spread"] = books["ask_0_price"] - books["bid_0_price"]
books["spread"].plot(title="spread")

# RSI


# Volatility
books["volatility"] = volatility(books["mid_price"].values)
