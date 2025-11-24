import sys
from pathlib import Path

BASE = Path().resolve()  # 這就是現在這個 notebook 所在的資料夾
print(BASE)              # 先印出來確認一下

sys.path.append(str(BASE / "Arsenal"))
sys.path.append(str(BASE / "StevenTricks"))


from user_lab.warehouse_manager import quick_inventory_summary

def demo_inventory():
    inv = quick_inventory_summary()
    print(inv)


from core_engine.app.stock_lab import quick_stock_vs_stock




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


from core_engine.data_access.twse_db import TwseDB
from core_engine.signals.basic.diff_tools import diff_for_stock

item = "三大法人買賣超日報"
subitem = "三大法人買賣超日報"

# 讀 raw table（只是示範）
db = TwseDB(item, subitem)
df_raw = db.load_table(decode_links=True)

# 各別算 diff
df_2330 = diff_for_stock(item, subitem, "2330", start="2024-01-01")
df_2317 = diff_for_stock(item, subitem, "2317", start="2024-01-01")

from core_engine.signals.basic.cumulative_tools import cumulative_tracker
from core_engine.signals.basic.signal_builder import build_signal_series

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


