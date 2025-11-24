import pandas as pd
from typing import Sequence
from core_engine.signals.basic.signal_builder import build_signal_series

def build_multi_indicator_signals(
    item: str,
    subitem: str,
    stock_id: str,
    value_cols: Sequence[str],
    start=None,
    end=None,
    mode: str = "diff",
) -> pd.DataFrame:
    """
    回傳一個 DataFrame：
        columns = 各種指標名稱
        index   = 日期
    """
    data = {}
    for col in value_cols:
        s = build_signal_series(item, subitem, stock_id, col, start, end, mode)
        data[col] = s

    df = pd.DataFrame(data).dropna(how="all")
    return df
