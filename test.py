import pandas as pd

df = pd.read_pickle(r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source/每月當日沖銷交易標的及統計/每月當日沖銷交易標的及統計_2025-11-30.pkl")

# 看看 date 是否唯一（一定不是）
print(df['date'].nunique(), len(df))

# 找出 date 重複的列
dup_date = df[df.duplicated(subset=['date'], keep=False)]
print(dup_date.head())

# 再看 date + stock_code 是否唯一
df['dup'] = df.duplicated(subset=['date', 'stock_code'], keep=False)
print(df['dup'].any())  # 理想情況應該是 False，代表這組 key 是唯一的


from core_engine.data_cleaning.twse import DEBUG_LAST_DF, DEBUG_LAST_CONTEXT

# 1. 確認這次 debug 的 item 是不是你要的那一張
DEBUG_LAST_CONTEXT.get("item"), DEBUG_LAST_CONTEXT.get("subitem")

# 2. 看欄位列表
list(DEBUG_LAST_DF.columns)

# 3. 看幾列 sample
DEBUG_LAST_DF.head(10)


from StevenTricks.core.convert_utils import _parse_single_date  # 名稱照你實際定義

a = _parse_single_date("110/06/01",mode=5)
b = _parse_single_date("113/12/05")


from pathlib import Path
import pandas as pd
import pprint

src_root = Path("/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source")
item_dir = src_root / "每日收盤行情"
p = item_dir / "每日收盤行情_2008-04-10.pkl"

raw = pd.read_pickle(p)  # 這個應該是 dict

print("keys =", list(raw.keys()))

# 看看 groups9 裡有什麼（通常會是一組 group 定義＋標題）
print("\n=== groups9 ===")
pprint.pp(raw.get("groups9"))

# 看 data7 / fields7 / subtitle7
print("\n=== subtitle7 ===")
print(raw.get("subtitle7"))
print("\nfields7 =", raw.get("fields7"))
print("\n第一列 data7：")
print(raw.get("data7")[0] if raw.get("data7") else None)

print("\n=== subtitle8 ===")
print(raw.get("subtitle8"))
print("\nfields8 =", raw.get("fields8"))
print("\n第一列 data8：")
print(raw.get("data8")[0] if raw.get("data8") else None)
