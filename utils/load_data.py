import lakeapi
import datetime
from time import time

from environment.env import OrderBook, Trade, MarketEvent


def load_book(max_depth=19):
    lakeapi.use_sample_data(anonymous_access=True)
    books = lakeapi.load_data(
        table="book",
        start=datetime.datetime(2022, 10, 1),
        end=datetime.datetime(2022, 10, 2),
        symbols=["BTC-USDT"],
        exchanges=["BINANCE"],
    )

    # data cleaning
    books.drop(columns=["exchange", "symbol", "sequence_number"], inplace=True)
    # convert datetime to float
    books["received_time"] = books["received_time"].apply(lambda x: x.timestamp())
    books["origin_time"] = books["origin_time"].apply(lambda x: x.timestamp())

    # check for missing values in the data
    print("Missing values for book : ", books.isnull().sum().sum(), "\n" + "-" * 50)
    # per column
    # print("missing values per colum :\n", books.isnull().sum())

    asks_price = books.filter(regex="ask_[0-9]+_price").values[:, :max_depth]
    asks_size = books.filter(regex="ask_[0-9]+_size").values[:, :max_depth]

    bids_price = books.filter(regex="bid_[0-9]+_price").values[:, :max_depth]
    bids_size = books.filter(regex="bid_[0-9]+_size").values[:, :max_depth]

    asks = [
        [(price, size) for price, size in zip(asks_price, asks_size)]
        for asks_price, asks_size in zip(asks_price, asks_size)
    ]
    bids = [
        [(price, size) for price, size in zip(bids_price, bids_size)]
        for bids_price, bids_size in zip(bids_price, bids_size)
    ]

    return list(
        OrderBook(*args)
        for args in zip(
            books.origin_time.values, books.received_time.values, asks, bids
        )
    )


def load_trades():

    lakeapi.use_sample_data(anonymous_access=True)
    trades = lakeapi.load_data(
        table="trades",
        start=datetime.datetime(2022, 10, 1),
        end=datetime.datetime(2022, 10, 2),
        symbols=["BTC-USDT"],
        exchanges=["BINANCE"],
    )

    trades = trades[["origin_time", "received_time", "side", "price", "quantity"]]

    trades["received_time"] = trades["received_time"].apply(lambda x: x.timestamp())
    trades["origin_time"] = trades["origin_time"].apply(lambda x: x.timestamp())

    # check for missing values in the data
    print("Missing values for trades : ", trades.isnull().sum().sum(), "\n" + "-" * 50)
    # per column
    # print("missing values per colum :\n", books.isnull().sum())

    return [Trade(*args) for args in trades.values]


def compute_market_event(books, trades):

    trades_dict = {(trade.exchange_ts, trade.receive_ts): trade for trade in trades}
    books_dict = {(book.exchange_ts, book.receive_ts): book for book in books}

    ts = sorted(trades_dict.keys() | books_dict.keys())

    return [
        MarketEvent(*key, books_dict.get(key, None), trades_dict.get(key, None))
        for key in ts
    ]


def load_data(max_depth=19):
    print("Loading book data...", end="\r")
    t1 = time()
    books = load_book(max_depth)
    t2 = time()
    print(f"Book data loaded in {t2-t1:.2f}s")

    print("Loading trade data...", end="\r")
    t1 = time()
    trades = load_trades()
    t2 = time()
    print(f"Trade data loaded in {t2-t1:.2f}s")

    print("Computing market events...", end="\r")
    t1 = time()
    md = compute_market_event(books, trades)
    t2 = time()
    print(f"Market events computed in {t2-t1:.2f}s")

    return md
