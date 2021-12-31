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
        df[f"{s}_RETURNS"] = np.log(df[s] / df[s].shift(1))

        # simulated total money
        df[f"{s}_STRATEGY_GAIN"] = 0

        # keeps track of how much money is invested
        inv = 0
        for i in df.index:
            if i == df.first_valid_index():
                df.loc[i, f"{s}_STRATEGY_GAIN"] = df.loc[i, s]
                inv += df.loc[i, s]
            else:
                last = df.index[df.index.get_loc(i) - 1]
                ret = (
                    1
                    if np.isnan(df.loc[last, f"{s}_RETURNS"])
                    else np.exp(df.loc[last, f"{s}_RETURNS"])
                )
                transactions = df.loc[i, f"{s}_TRANSACTIONS"] * df.loc[i, s]

                df.loc[i, f"{s}_STRATEGY_GAIN"] = (
                    ret * df.loc[last, f"{s}_STRATEGY_GAIN"] + transactions
                )
                inv += transactions

        simulation_results[s] = {
            "base returns": df.iloc[-1][s] / df.iloc[0][s],
            "amount invested": inv,
            "strategy returns": df.iloc[-1][f"{s}_STRATEGY_GAIN"] / inv,
        }
    return simulation_results


def add_dca_transactions(df, stocks):
    """Adds dollar cost averaging strategy transactions"""
    for s in stocks:
        df[f"{s}_TRANSACTIONS"] = 0
        df.loc[df.index.day < 10, f"{s}_TRANSACTIONS"] = 1
        df[f"{s}_TRANSACTIONS"] = df[f"{s}_TRANSACTIONS"].diff().clip(lower=0)
        df.loc[df.index[0], f"{s}_TRANSACTIONS"] = 1
