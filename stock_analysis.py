import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.append(str(BASE / "Arsenal"))
sys.path.append(str(BASE / "StevenTricks"))

from app.stock_lab import quick_stock_vs_stock,quick_indicators_for_stock

from app.stock_lab import quick_stock_vs_stock

df_two = quick_stock_vs_stock(
    stock_a="2330",
    stock_b="2317",
    value_col="外資買賣超股數",
    start="2024-01-01",
    end="2024-06-30",
    mode="diff",  # raw / previous / diff / diff_percent
)

print(df_two.head())
# columns 大概會是：
# date, 代號_A, value_A, 代號_B, value_B 或類似結構（看你現在的實作）


from data_access.twse_db import TwseDB
from signals.basic.diff_tools import diff_for_stock

item = "三大法人買賣超日報"
subitem = "三大法人買賣超日報"

# 讀 raw table（只是示範）
db = TwseDB(item, subitem)
df_raw = db.load_table(decode_links=True)

# 各別算 diff
df_2330 = diff_for_stock(item, subitem, "2330", start="2024-01-01")
df_2317 = diff_for_stock(item, subitem, "2317", start="2024-01-01")

from signals.basic.cumulative_tools import cumulative_tracker
from signals.basic.signal_builder import build_signal_series

s = build_signal_series(
    item="三大法人買賣超日報",
    subitem="三大法人買賣超日報",
    stock_id="2330",
    value_col="外資買賣超股數",
    start="2024-01-01",
    mode="diff",
)

date_hit, cum, reason = cumulative_tracker(
    s,
    start_date="2024-01-01",
    threshold=1_000_000,
)

print(date_hit, cum, reason)


