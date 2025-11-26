"""
warehouse_report.py

倉庫管理員的「報表入口」。

定位：給你一個一鍵檢查流程：
    - 整體 cleaned 狀況
    - source vs cleaned 的缺漏
    - （若有）清理 log 的摘要
"""

from __future__ import annotations

import argparse
from datetime import date
import pandas as pd

from warehouse_manager import twse_inventory as inv


def format_title(title: str) -> str:
    bar = "=" * len(title)
    return f"{title}\n{bar}"


def run_report(show_log: bool = True) -> None:
    today = date.today()

    cleaned_detail, item_summary, relation_status = inv.quick_inventory_summary(today=today)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print(format_title(f"TWSE 倉庫總覽（截至 {today}）"))

    # 1. item 層級 summary
    print("\n[1] Cleaned 依 item 彙總：")
    if item_summary.empty:
        print("  （尚無 cleaned 資料）")
    else:
        print(item_summary.to_string(index=False))

    # 2. source vs cleaned 缺漏
    print("\n[2] Source vs Cleaned 缺漏檢查：")
    missing = inv.find_missing_cleaned(today=today)
    orphan = inv.find_orphan_cleaned(today=today)

    if missing.empty and orphan.empty:
        print("  ✔ source / cleaned 結構一致，沒有缺漏或孤兒檔。")
    else:
        if not missing.empty:
            print("\n  2.1 source 有但 cleaned 沒有（缺少清理或路徑設定）：")
            print(missing[["item", "subitem", "source_mtime", "source_status"]].to_string(index=False))
        if not orphan.empty:
            print("\n  2.2 cleaned 有但 source 沒有（可能為舊格式或路徑未設定）：")
            print(orphan[["item", "subitem", "cleaned_mtime", "cleaned_status"]].to_string(index=False))

    # 3. （選配）清理 log 彙總
    if show_log:
        print("\n[3] 清理 log 摘要（若 log.pkl 存在）：")
        log_summary = inv.summarize_cleaning_log()
        if log_summary.empty:
            print("  （找不到 log.pkl 或內容為空）")
        else:
            # 只印前 50 列，避免過長
            print(log_summary.head(50).to_string(index=False))
            if len(log_summary) > 50:
                print(f"\n  ... 共 {len(log_summary)} 筆，只顯示前 50 筆。")


def main():
    parser = argparse.ArgumentParser(description="TWSE 倉庫管理員報表")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="不要顯示清理 log 摘要",
    )
    args = parser.parse_args()
    run_report(show_log=not args.no_log)


if __name__ == "__main__":
    main()
