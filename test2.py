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

# 假設 df 已經準備好，且包含 Funding_Date_yymm, Funding_Amt 等欄位
BASE_PERIOD = 202510
COMP_PERIOD = 202511

result_all = run_driver_tree_change(
    df=df,
    base_period=BASE_PERIOD,
    comp_period=COMP_PERIOD,
    # dims=None 代表用 DIM_ROLE 中所有非 time 欄位
    dims=None,
    target_col="Funding_Amt",
    time_col="Funding_Date_yymm",
    max_depth=3,
    min_node_share=0.05,
    top_k=5,
    split_policy="best_overall",
)

nodes_df = result_all["nodes_df"]
root = result_all["root"]

print("=== 整體變動摘要 ===")
print(root.summary_zh)

print("=== 節點列表（前幾筆） ===")
print(nodes_df.head())



focus_dims = [
    "Property_Location_Flag",
    "Tenor_Flag",
    "Grace_Length_Flag",
]

result_focus = run_driver_tree_change(
    df=df,
    base_period=BASE_PERIOD,
    comp_period=COMP_PERIOD,
    dims=focus_dims,          # ✅ 只用這幾個欄位拆解
    target_col="Funding_Amt",
    time_col="Funding_Date_yymm",
    max_depth=3,
    min_node_share=0.05,
    top_k=5,
    split_policy="best_overall",
)

print("=== 指定三個欄位拆解的 root 摘要 ===")
print(result_focus["root"].summary_zh)

print("=== root 節點使用的 split_dim ===")
print(result_focus["root"].split_dim)

print("=== root 對應的 pivot（看哪個類別影響最大） ===")
root_pivot = result_focus["pivots"][result_focus["root"].node_id]
print(root_pivot.sort_values("abs_delta", ascending=False).head(10))
