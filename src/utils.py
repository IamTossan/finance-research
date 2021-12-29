import pandas as pd
import pandas_datareader.data as reader
import datetime as dt
from os.path import exists

def get_data(path, stocks, fetch=False):
    end = dt.datetime.now()
    start = dt.date(end.year - 5, end.month, end.day)

    if fetch or not exists(path):
        df = reader.get_data_yahoo(stocks, start, end)["Adj Close"].dropna()
        df.to_csv(path)
    else:
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
    return df