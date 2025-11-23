import pandas as pd
from signals.basic.signal_builder import build_signal_series
from signals.basic.cumulative_tools import cumulative_tracker

def compare_two_stocks_signal(
    item: str,
    subitem: str,
    stock_a: str,
    stock_b: str,
    value_col: str,
    start=None,
    end=None,
    mode: str = "diff",
) -> pd.DataFrame:
    """
    回傳一個 DataFrame，包含兩檔個股在同一時間軸上的 signal。
    """
    sa = build_signal_series(item, subitem, stock_a, value_col, start, end, mode)
    sb = build_signal_series(item, subitem, stock_b, value_col, start, end, mode)

    df = pd.DataFrame({
        stock_a: sa,
        stock_b: sb,
    }).dropna(how="all")

    return df

def compare_cumulative_threshold(
    series_a: pd.Series,
    series_b: pd.Series,
    threshold: float,
    start_date: str,
):
    """
    用 cumulative_tracker 比較兩檔個股達到同一累積門檻的日期。
    """
    da, ca, ra = cumulative_tracker(series_a, start_date=start_date, threshold=threshold)
    db, cb, rb = cumulative_tracker(series_b, start_date=start_date, threshold=threshold)
    return {
        "A": {"date": da, "cum": ca, "reason": ra},
        "B": {"date": db, "cum": cb, "reason": rb},
    }
