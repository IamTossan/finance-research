import numpy as np
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


def get_return_simulation(df, stocks):
    """Returns investment simulation metrics
    df: pd.Dataframe
        df[keyof stocks]: number[]
        df[f"{keyof stocks}_TRANSACTIONS"]: 0|1
    stocks: str[]
    """
    simulation_results = {}

    for s in stocks:
        # base returns
        df[f"{s}_RETURNS"] = (df[s] / df[s].shift(1)).fillna(1)

        # simulated total money
        df[f"{s}_STRATEGY_GAIN"] = 0

        # keeps track of how much money is invested
        inv = 0

        for tr in df[df[f"{s}_TRANSACTIONS"] == 1].index:
            ser = pd.Series(0, index=df.index)

            cur_share_value = df.loc[tr, s]
            ser[ser.index >= tr] = cur_share_value
            inv += cur_share_value

            ret = df[f"{s}_RETURNS"].copy()
            ret[ret.index <= tr] = 1

            ser *= ret.cumprod()
            df[f"{s}_STRATEGY_GAIN"] += ser

        simulation_results[s] = {
            "base returns": df.iloc[-1][s] / df.iloc[0][s],
            "amount invested": inv,
            "strategy returns": df.iloc[-1][f"{s}_STRATEGY_GAIN"] / inv,
        }
    return simulation_results
