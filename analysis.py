import lakeapi
import datetime
import numpy as np
import matplotlib.pyplot as plt

from utils.features import inventory_ratio, volatility, RSI, book_imbalance

lakeapi.use_sample_data(anonymous_access=True)

books = lakeapi.load_data(
    table="book",
    start=datetime.datetime(2022, 10, 1),
    end=datetime.datetime(2022, 10, 2),
    symbols=["BTC-USDT"],
    exchanges=["BINANCE"],
)

books["mid_price"] = (books["ask_0_price"] + books["bid_0_price"]) / 2


# Book Imbalance

asks_size = books.filter(regex="ask_[0-9]+_size")
bids_size = books.filter(regex="bid_[0-9]+_size")

bids_size = np.sum(bids_size, axis=1)
asks_size = np.sum(asks_size, axis=1)

books["imb"] = (asks_size - bids_size) / (asks_size + bids_size)
books["target_100"] = books["mid_price"].diff(100) / books["mid_price"]
books["target_1000"] = books["mid_price"].diff(1000) / books["mid_price"]
books["target_10000"] = books["mid_price"].diff(10000) / books["mid_price"]

books.plot.scatter(x="imb", y="target_100", alpha=0.1, title="imb vs target_100")


books["imb"].rolling(30000).corr(books["target_100"]).dropna().plot()


#  reg liner model
from statsmodels.api import OLS

data = books.dropna(subset=["imb", "target_100", "target_1000", "target_10000"])
X = data["imb"].values.reshape(-1, 1)

R2 = []


def r2_out_of_sample(y, X):
    model = OLS(y, X)
    results = model.fit()
    OutY_Predicted = results.predict(X)

    return results.rsquared  # y.corr(OutY_Predicted) ** 2


for target in ["target_100", "target_1000", "target_10000"]:
    y = data[target]
    R2.append(r2_out_of_sample(y, X))

plt.scatter(["target_100", "target_1000", "target_10000"], R2)
plt.title("R2 for different targets vs imb")


# spread
books["spread"] = books["ask_0_price"] - books["bid_0_price"]
books["spread"].plot(title="spread")

# RSI


# Volatility
books["volatility"] = volatility(books["mid_price"].values)
