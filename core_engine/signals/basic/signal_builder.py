import pandas as pd
from typing import Optional
from core_engine.data_access.twse_db import TwseDB
from core_engine.signals.basic.diff_tools import diff_previous

"""
「我給你 item / subitem / stock_id / value_col / mode，請你回給我一條 index=日期、值是我要的信號的 Series。」
"""

def build_signal_series(
    item: str,
    subitem: str,
    stock_id: str,
    value_col: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    mode: str = "diff",  # "raw" / "previous" / "diff" / "diff_percent"
) -> pd.Series:
    """
    回傳 index=date 的單一數列，用來做比較。
    """
    db = TwseDB(item, subitem)
    base_df = db.get_stock_df(stock_id, start=start, end=end)

    # 加上 diff 資訊
    df = diff_previous(base_df, item=item, subitem=subitem, percent=True)

    col_map = {
        "raw": value_col,
        "previous": f"previous_{value_col}",
        "diff": f"diff_{value_col}",
        "diff_percent": f"diff_percent_{value_col}",
    }
    target_col = col_map[mode]

    if target_col not in df.columns:
        raise KeyError(f"欄位 {target_col} 不存在於 diff_previous 結果中。")

    s = df.set_index("date")[target_col].sort_index()
    return s
