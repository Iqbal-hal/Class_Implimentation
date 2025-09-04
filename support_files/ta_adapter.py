"""Small adapter that exposes a subset of the TA-Lib API but falls back to
pandas_ta when the C-extension `talib` is not available.

It provides the functions used across the project: EMA, RSI, MACD, BBANDS,
ROC, ADX, PLUS_DI, MINUS_DI, ATR.
"""
from __future__ import annotations

try:
    import talib as _talib
    # When the real talib is available, expose its functions directly
    EMA = _talib.EMA
    RSI = _talib.RSI
    MACD = _talib.MACD
    BBANDS = _talib.BBANDS
    ROC = _talib.ROC
    ADX = _talib.ADX
    PLUS_DI = _talib.PLUS_DI
    MINUS_DI = _talib.MINUS_DI
    ATR = _talib.ATR
    _backend = 'talib'
except Exception:
    _backend = 'pandas_ta'
    import pandas as pd
    try:
        import pandas_ta as pta
    except Exception:  # pragma: no cover - user should install pandas_ta in this case
        raise ImportError("Neither talib nor pandas_ta are available; please install one of them.")

    def EMA(series: pd.Series, timeperiod: int = 30) -> pd.Series:
        return pta.ema(series, length=timeperiod)

    def RSI(series: pd.Series, timeperiod: int = 14) -> pd.Series:
        return pta.rsi(series, length=timeperiod)

    def MACD(series: pd.Series, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
        # MACD = EMA(fast) - EMA(slow); signal = EMA(MACD, signalperiod); hist = MACD - signal
        macd_line = EMA(series, fastperiod) - EMA(series, slowperiod)
        signal = EMA(macd_line, signalperiod)
        hist = macd_line - signal
        return macd_line, signal, hist

    def BBANDS(series: pd.Series, timeperiod: int = 5, nbdevup: float = 2.0, nbdevdn: float = 2.0, matype: int = 0):
        middle = series.rolling(window=timeperiod, min_periods=1).mean()
        std = series.rolling(window=timeperiod, min_periods=1).std()
        upper = middle + nbdevup * std
        lower = middle - nbdevdn * std
        return upper, middle, lower

    def ROC(series: pd.Series, timeperiod: int = 10) -> pd.Series:
        return (series - series.shift(timeperiod)) / series.shift(timeperiod) * 100.0

    def ADX(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
        df = pta.adx(high=high, low=low, close=close, length=timeperiod)
        key = f'ADX_{timeperiod}'
        return df[key] if key in df else df.iloc[:, 0]

    def PLUS_DI(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
        df = pta.adx(high=high, low=low, close=close, length=timeperiod)
        key = f'DMP_{timeperiod}'
        return df[key] if key in df else df.iloc[:, 1]

    def MINUS_DI(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
        df = pta.adx(high=high, low=low, close=close, length=timeperiod)
        key = f'DMN_{timeperiod}'
        return df[key] if key in df else df.iloc[:, 2] if df.shape[1] > 2 else df.iloc[:, -1]

    def ATR(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
        df = pta.atr(high=high, low=low, close=close, length=timeperiod)
        key = f'ATR_{timeperiod}'
        return df[key] if key in df else df.iloc[:, 0]

__all__ = [
    'EMA', 'RSI', 'MACD', 'BBANDS', 'ROC', 'ADX', 'PLUS_DI', 'MINUS_DI', 'ATR', '_backend'
]
