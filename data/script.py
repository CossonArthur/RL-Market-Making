#Data coming from https://crypto-lake.com/free-data/


import lakeapi
import datetime

lakeapi.use_sample_data(anonymous_access = True)
books = lakeapi.load_data(
    table="book",
    start=datetime.datetime(2022, 10, 1),
    end=datetime.datetime(2022, 10, 2),
    symbols=["BTC-USDT"],
    exchanges=["BINANCE"],
)
books.set_index('received_time', inplace = True)
books.drop(columns = ['exchange', 'symbol','origin_time', 'sequence_number'], inplace = True)

