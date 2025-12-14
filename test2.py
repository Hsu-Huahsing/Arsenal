# -*- coding: utf-8 -*-
"""
測試 StevenTricks.analysis.driver_tree 的簡單範例
使用檔案：/Users/stevenhsu/Downloads/202511新貸.xlsx
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------
# 1. 把 StevenTricks 專案加到 sys.path（若你已經 pip install 就不需要）
# ---------------------------------------------------------
PROJECT_ROOT = "/Users/stevenhsu/programming/StevenTricks"  # TODO: 改成你的實際路徑
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, "StevenTricks")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from StevenTricks.analysis.driver_tree import run_driver_tree_change

# ---------------------------------------------------------
# 2. 讀取 Excel 檔
# ---------------------------------------------------------
EXCEL_PATH = "/Users/stevenhsu/Downloads/202511新貸.xlsx"  # TODO: 若路徑不同請修改

df = pd.read_excel(EXCEL_PATH)

print("原始資料筆數與欄位：")
print(df.shape)
print(df.columns.tolist())

# 看一下有哪些月份，確認 base / comp 可以選什麼
print("\nFunding_Date_yymm 的唯一值：")
print(df["Funding_Date_yymm"].value_counts())

# ---------------------------------------------------------
# 3. 設定 base_period / comp_period
#    假設你要比較 202510 → 202511（自己確認真的有這兩個值）
# ---------------------------------------------------------
BASE_PERIOD = 202510
COMP_PERIOD = 202511

# ---------------------------------------------------------
# 4. 要納入 Driver Tree 的維度欄位
#    -> 若某欄在 df 裡不存在，driver_tree 內部會自動略過
# ---------------------------------------------------------
dims = [
    "Product_Flag_new",
    "Property_Location_Flag",
    "Tenor_Flag",
    "Grace_Length_Flag",
    "OLTV_Flag",
    "Cust_Flag",
    "Investor_Flag",
    "Acct_Type_Code",
    "Int_Category_Code",
    "Batch_Flag",
    "special_flag",
    "Public_Flag2024",
    "cb_Investor_flag",
]

# ---------------------------------------------------------
# 5. 跑 Driver Tree
# ---------------------------------------------------------
result = run_driver_tree_change(
    df=df,
    base_period=BASE_PERIOD,
    comp_period=COMP_PERIOD,
    dims=dims,                     # 若想讓它自動用 DIM_ROLE 裡能找到的欄位，也可以填 None
    target_col="Funding_Amt",
    time_col="Funding_Date_yymm",
    max_depth=3,
    min_node_share=0.05,
    top_k=5,
)

root = result["root"]
nodes_df = result["nodes_df"]
pivots = result["pivots"]

print("\n=== 節點摘要（前 20 列） ===")
print(nodes_df.head(20))

print("\n=== 根節點資訊 ===")
print("root node_id:", root.node_id)
print("root path:", root.path)
print("root amt_base:", root.amt_base)
print("root amt_comp:", root.amt_comp)
print("root delta_amt:", root.delta_amt)
print("root delta_share:", root.delta_share)
print("root summary_zh:", root.summary_zh)

TARGET_PRODUCT = "02.融資型"

df_loan = df[df["Product_Flag_new"] == TARGET_PRODUCT].copy()

dims_loan = [
    d for d in dims
    if d != "Product_Flag_new" and d in df_loan.columns
]

result_loan = run_driver_tree_change(
    df=df_loan,
    base_period=BASE_PERIOD,
    comp_period=COMP_PERIOD,
    dims=dims_loan,
    target_col="Funding_Amt",
    time_col="Funding_Date_yymm",
    max_depth=5,         # ✅ 想更細就拉高
    min_node_share=0.02, # ✅ 融資型內部：2%門檻通常比較合理
    top_k=10,            # ✅ 類別多就調大
)

root_loan = result_loan["root"]
nodes_loan = result_loan["nodes_df"]
pivots_loan = result_loan["pivots"]

print(root_loan.summary_zh)

