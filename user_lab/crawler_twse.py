# -*- coding: utf-8 -*-
"""
user_lab/crawler_twse.py

用途：
    - 從 user_lab 這邊操作 TWSE 爬蟲。
    - 提供：
        run_update()        → 等同於 python -m crawler.twse --update
        run_symbol(...)     → 抓單一股票最近 N 日資料（twse.py 的 main 包起來）

使用方式：
    1) 在 IPython / Jupyter 裡：
        %run -i path/to/Arsenal/user_lab/crawler_twse.py
        run_update()
        # 或
        run_symbol("2330", days=90)

    2) 直接當腳本跑：
        python user_lab/crawler_twse.py --update
        python user_lab/crawler_twse.py --symbol 2330 --days 90
"""

from __future__ import annotations
from pathlib import Path
import sys
import argparse

# ---------------------------------------------------------------------------
# 0. sys.path 設定：讓 core_engine 可以被 import（跟 warehouse_twse 一致）
# ---------------------------------------------------------------------------

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Arsenal
else:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

ST_ROOT = PROJECT_ROOT.parent / "StevenTricks"
if ST_ROOT.exists() and str(ST_ROOT) not in sys.path:
    sys.path.append(str(ST_ROOT))

# ---------------------------------------------------------------------------
# 1. 匯入 twse crawler 的 main
# ---------------------------------------------------------------------------

from core_engine.crawler.twse import main as twse_main


# ---------------------------------------------------------------------------
# 2. 封裝成你在互動環境好叫的函式
# ---------------------------------------------------------------------------

def run_update() -> None:
    """
    等同於：python -m crawler.twse --update
    從 TWSE_COLLECTION 設定的所有 item 去補齊/更新資料。
    """
    twse_main(["--update"])


def run_symbol(symbol: str, days: int = 90) -> None:
    """
    等同於：python -m crawler.twse --symbol <代號> --days <天數>

    參數
    ----
    symbol : 股票代號，例如 "2330"
    days   : 要抓幾天，預設 90
    """
    twse_main(["--symbol", str(symbol), "--days", str(days)])


# ---------------------------------------------------------------------------
# 3. 直接當腳本跑時，支援簡單的 CLI 參數
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TWSE 爬蟲（user_lab 入口）")
    parser.add_argument(
        "--update",
        action="store_true",
        help="更新所有 TWSE collection 定義的項目",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="指定股票代號，例如 2330",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="搭配 --symbol 使用，預設 90 天",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if args.update:
        run_update()
    elif args.symbol:
        run_symbol(args.symbol, days=args.days)
    else:
        print("請使用 --update 或 --symbol / --days，例如：")
        print("  python crawler_twse.py --update")
        print("  python crawler_twse.py --symbol 2330 --days 90")
