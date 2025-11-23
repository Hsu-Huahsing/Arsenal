from Analysis.stock_vs_stock import compare_two_stocks_signal
from Analysis.indicator_vs_indicator import build_multi_indicator_signals

def quick_stock_vs_stock(
    stock_a: str,
    stock_b: str,
    value_col: str,
    start=None,
    end=None,
    mode: str = "diff",
    item: str = "三大法人買賣超日報",
    subitem: str = "三大法人買賣超日報",
):
    """
    給你最常用的操作：
    - 固定 item/subitem
    - 只要丟兩檔股票 + 一個指標
    """
    return compare_two_stocks_signal(
        item=item,
        subitem=subitem,
        stock_a=stock_a,
        stock_b=stock_b,
        value_col=value_col,
        start=start,
        end=end,
        mode=mode,
    )

def quick_indicators_for_stock(
    stock_id: str,
    value_cols,
    start=None,
    end=None,
    mode: str = "diff",
    item: str = "三大法人買賣超日報",
    subitem: str = "三大法人買賣超日報",
):
    """
    給你最常用的操作：
    - 一支股票 + 多個指標
    """
    return build_multi_indicator_signals(
        item=item,
        subitem=subitem,
        stock_id=stock_id,
        value_cols=value_cols,
        start=start,
        end=end,
        mode=mode,
    )
