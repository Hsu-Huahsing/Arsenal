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
