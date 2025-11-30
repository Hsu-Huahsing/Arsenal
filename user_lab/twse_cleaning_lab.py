# -*- coding: utf-8 -*-
"""
user_lab/twse_cleaning_lab.py

用途：
    - 從 user_lab 這邊操作 TWSE cleaned 檔的整理流程。
    - 封裝 core_engine.data_cleaning.twse.process_twse_data：
        1) run_cloud_staging_clean()  → 你現在主要使用的雲端 + staging 模式
        2) run_cloud_clean()          → 完全雲端模式
        3) run_local_clean()          → 完全本機模式（沒有雲端路徑時）

使用方式：

    1) 在 IPython / PyCharm console 裡：

        %run -i path/to/Arsenal/user_lab/twse_cleaning_lab.py

        # 然後在互動環境裡手動下：
        run_cloud_staging_clean()
        # 或：
        run_cloud_clean()
        run_local_clean(batch_size=1000)

    2) 直接當腳本跑（Terminal）：

        python user_lab/twse_cleaning_lab.py --mode cloud_staging --batch-size 500 --bucket year
        python user_lab/twse_cleaning_lab.py --mode cloud
        python user_lab/twse_cleaning_lab.py --mode local --batch-size 500
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

# ---------------------------------------------------------------------------
# 0. sys.path 設定：讓 core_engine 可以被 import（跟 warehouse_twse / twse_crawler_lab 一致）
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
# 1. 匯入 TWSE 清理主程式
# ---------------------------------------------------------------------------

from core_engine.data_cleaning.twse import process_twse_data


# ---------------------------------------------------------------------------
# 2. 給互動環境用的 wrapper
# ---------------------------------------------------------------------------

def run_cloud_staging_clean(
    batch_size: int = 500,
    bucket_mode: str = "year",
) -> None:
    """
    雲端 + 本機暫存（staging）模式：
        - 從雲端抓下來一批檔案到 staging
        - 做 cleaned
        - 再同步回雲端

    這是你現在 test.py 原本的預設行為。
    """
    process_twse_data(
        storage_mode="cloud_staging",
        batch_size=batch_size,
        bucket_mode=bucket_mode,
    )


def run_cloud_clean(
    bucket_mode: str = "year",
) -> None:
    """
    完全雲端模式：
        - 不用 staging，本機不存中間檔。
    """
    process_twse_data(
        storage_mode="cloud",
        batch_size=None,   # cloud 模式用不到
        bucket_mode=bucket_mode,
    )


def run_local_clean(
    batch_size: int = 500,
    bucket_mode: str = "year",
) -> None:
    """
    完全本機模式：
        - 所有 source / cleaned 都在本機路徑底下，不碰雲端。
    """
    process_twse_data(
        storage_mode="local",
        batch_size=batch_size,
        bucket_mode=bucket_mode,
    )


# ---------------------------------------------------------------------------
# 3. 當作腳本直接執行時的 CLI 入口
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TWSE cleaned 資料整理（user_lab 入口）")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cloud_staging", "cloud", "local"],
        default="cloud_staging",
        help="清理模式：cloud_staging / cloud / local（預設 cloud_staging）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="每批次處理幾個檔案（僅 cloud_staging / local 模式有用；預設 500）",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="year",
        help="分桶模式，例如 'year'（預設）",
    )
    return parser.parse_args(argv)


def _run_from_cli(args: argparse.Namespace) -> None:
    if args.mode == "cloud_staging":
        run_cloud_staging_clean(batch_size=args.batch_size, bucket_mode=args.bucket)
    elif args.mode == "cloud":
        run_cloud_clean(bucket_mode=args.bucket)
    elif args.mode == "local":
        run_local_clean(batch_size=args.batch_size, bucket_mode=args.bucket)
    else:
        raise ValueError(f"未知的 mode：{args.mode!r}")


if __name__ == "__main__":
    _args = _parse_args()
    _run_from_cli(_args)
