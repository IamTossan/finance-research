import numpy as np
import pandas as pd


def dca(df, stocks):
    """Adds dollar cost averaging strategy transactions"""
    for s in stocks:
        df[f"{s}_TRANSACTIONS"] = 0
        df.loc[df.index.day < 10, f"{s}_TRANSACTIONS"] = 1
        df[f"{s}_TRANSACTIONS"] = df[f"{s}_TRANSACTIONS"].diff().clip(lower=0)
        df.loc[df.index[0], f"{s}_TRANSACTIONS"] = 1


def sma(df, stocks, sma1, sma2):
    df = pd.concat(
        [
            df[stocks],
            df[stocks].rolling(sma1).mean().add_suffix(f"_SMA{sma1}"),
            df[stocks].rolling(sma2).mean().add_suffix(f"_SMA{sma2}"),
        ],
        axis=1,
    )

    for s in stocks:
        df[f"{s}_POSITION"] = np.where(
            df[f"{s}_SMA{sma1}"] > df[f"{s}_SMA{sma2}"], 1, 0
        )
        df[f"{s}_TRANSACTIONS"] = df[f"{s}_POSITION"].diff().clip(lower=0)
        df.loc[df.index[0], f"{s}_TRANSACTIONS"] = 1

    return df


def ema(df, stocks, ema1, ema2):
    df = pd.concat(
        [
            df[stocks],
            df[stocks].ewm(ema1, adjust=False).mean().add_suffix(f"_EMA{ema1}"),
            df[stocks].ewm(ema2, adjust=False).mean().add_suffix(f"_EMA{ema2}"),
        ],
        axis=1,
    )

    for s in stocks:
        df[f"{s}_POSITION"] = np.where(
            df[f"{s}_EMA{ema1}"] > df[f"{s}_EMA{ema2}"], 1, 0
        )
        df[f"{s}_TRANSACTIONS"] = df[f"{s}_POSITION"].diff().clip(lower=0)
        df.loc[df.index[0], f"{s}_TRANSACTIONS"] = 1

    return df


def rsi(df, stocks, periods=14, threshold=30, ema=True):
    close_delta = df[stocks].diff()

    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema:
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    else:
        ma_up = up.rolling(periods).mean()
        ma_down = down.rolling(periods).mean()

    rs = ma_up / ma_down
    rsi = (100 - (100 / (1 + rs))).add_suffix("_RSI")

    dff = pd.DataFrame(rsi)

    for s in stocks:

        dff[f"{s}_POSITION"] = np.where(dff[f"{s}_RSI"] < threshold, 0, 1)

        dff[f"{s}_TRANSACTIONS"] = dff[f"{s}_POSITION"].diff().clip(lower=0)
        dff.loc[dff.index[0], f"{s}_TRANSACTIONS"] = 1

    return dff

def bollinger_bands(df, stocks, periods=20):
    tp = (df['Close'] + df['Low'] + df['High']) / 3
    std = tp.rolling(periods).std(ddof=0)
    ma = tp.rolling(periods).mean()
    bolu = ma + 2 * std
    bold = ma - 2 * std

    dff = pd.concat([
        df['Close'],
        bolu.add_suffix('_BOLU'),
        bold.add_suffix('_BOLD')
    ], axis=1)

    for s in stocks:
        dff[f"{s}_POSITION"] = np.where(
            dff[s] <= dff[f"{s}_BOLD"], 1, 0
        )
        dff[f"{s}_TRANSACTIONS"] = dff[f"{s}_POSITION"].diff().clip(lower=0)
        dff.loc[dff.index[0], f"{s}_TRANSACTIONS"] = 1
    
    return dff

def stochastic_oscillator(df, stocks, k_period=14, d_period=3):
    n_high = df['High'].rolling(k_period).max()
    n_low = df['Low'].rolling(k_period).min()

    k = (df['Close'] - n_low) * 100 / (n_high - n_low)
    d = k.rolling(d_period).mean()

    dff = pd.concat([
        df['Close'],
        k.add_suffix('_%K'),
        d.add_suffix('_%D')
    ], axis=1)

    for s in stocks:
        dff[f"{s}_POSITION"] = np.where((dff[f"{s}_%K"] < 20) & (dff[f"{s}_%D"] < 20) & (dff[f"{s}_%K"] > dff[f"{s}_%D"]), 1, 0)

        dff[f"{s}_TRANSACTIONS"] = dff[f"{s}_POSITION"].diff().clip(lower=0)
        dff.loc[dff.index[0], f"{s}_TRANSACTIONS"] = 1

    return dff
