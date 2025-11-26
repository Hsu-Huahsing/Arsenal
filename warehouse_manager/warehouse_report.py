"""
舊的 warehouse_report.py 的精簡版

實際邏輯已整併到 warehouse_manager.twse_inventory.run_report；
本檔僅保留一個薄封裝，方便舊習慣或舊腳本沿用。
"""

from __future__ import annotations

import argparse

from warehouse_manager.twse_inventory import run_report


def main() -> None:
    parser = argparse.ArgumentParser(description="TWSE 倉庫管理員報表（wrapper）")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="不要顯示清理 log / error log 摘要",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="顯示 cleaned / relation 的明細（前 N 列）",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=30,
        help="明細列數上限（預設 30）",
    )
    args = parser.parse_args()

    run_report(
        today=None,
        show_log=not args.no_log,
        show_detail=args.detail,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
