import numpy as np
import pandas as pd
import pandas_datareader.data as reader
import datetime as dt
import os

symbols = {
    "stocks": {
        "symbols": ["MFED.PA", "PE500.PA", "EN.PA", "TTE.PA"],
        "path": "../data/stocks.csv"
    },
    "cryptos": {
        "symbols": ['BTC-EUR', 'ETH-EUR', 'DOGE-EUR', 'ADA-EUR', 'SHIB-EUR'],
        "path": "../data/cryptos.csv"
    },
    "cac40": {
        "symbols": [
            "AC.PA",
            "ACA.PA",
            "AI.PA",
            "AIR.PA",
            "ALO.PA",
            "ATO.PA",
            "BN.PA",
            "BNP.PA",
            "CA.PA",
            "CAP.PA",

            "CS.PA",
            "DG.PA",
            "DSY.PA",
            "EN.PA",
            "ENGI.PA",
            "GLE.PA",
            "HO.PA",
            "KER.PA",
            "LR.PA",
            "MC.PA",

            "ML.PA",
            "OR.PA",
            "ORA.PA",
            "PUB.PA",
            "RI.PA",
            "RMS.PA",
            "RNO.PA",
            "SAF.PA",
            "SAN.PA",
            "SGO.PA",

            "STLA.PA",
            "STM.PA",
            "SU.PA",
            "SW.PA",
            "TEP.PA",
            "TTE.PA",
            "URW.AS",
            "VIE.PA",
            "VIV.PA",
            "WLN.PA",
        ],
        "path": "../data/cac40.csv"
    },
}

def get_data(kind, fetch=False, columns=["Adj Close"]):
    if kind not in symbols.keys():
        raise KeyError(f'{kind} is not a valid kind. try {"".join(symbols.keys())}')

    end = dt.datetime.now()
    start = dt.date(end.year - 5, end.month, end.day)
    path = os.path.join(os.path.dirname(__file__), symbols[kind]["path"])

    if fetch or not os.path.exists(path):
        df = reader.get_data_yahoo(symbols[kind]["symbols"], start, end)
        df.to_csv(path)
    else:
        df = pd.read_csv(path, parse_dates=[0], index_col=0, header=[0, 1])

    if len(columns) == 1:
        return df[columns[0]].dropna(), symbols[kind]["symbols"]
    return df[columns].dropna(), symbols[kind]["symbols"]


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
