# -*- coding: utf-8 -*-
"""
user_lab/cleaning_twse.py

用途（使用者視角）：
    - 提供「一行就能叫用」的 TWSE cleaned 清理入口，不再出現任何 CLI / IPython 教學。
    - 真正的清理邏輯全部在工具層 core_engine.data_cleaning.twse 裡面實作。
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Optional, Union

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Arsenal
else:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

STEVEN_TRICKS_ROOT = PROJECT_ROOT.parent / "StevenTricks"
if STEVEN_TRICKS_ROOT.exists() and str(STEVEN_TRICKS_ROOT) not in sys.path:
    sys.path.append(str(STEVEN_TRICKS_ROOT))

from core_engine.data_cleaning.twse import process_twse_data


def _coerce_cols(cols: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    if cols is None:
        return None
    if isinstance(cols, str):
        s = cols.strip()
        return [s] if s else None
    out = [str(x).strip() for x in cols if str(x).strip()]
    return out if out else None


def run_cloud_staging_clean(
    *,
    cols: Optional[Union[str, List[str]]] = None,
    batch_size: int = 500,
    bucket_mode: str = "year",
    update_old_non_daily: bool = False,
) -> None:
    process_twse_data(
        cols=_coerce_cols(cols),
        storage_mode="cloud_staging",
        batch_size=batch_size,
        bucket_mode=bucket_mode,
        update_old_non_daily=update_old_non_daily,
    )


def run_cloud_clean(
    *,
    cols: Optional[Union[str, List[str]]] = None,
    bucket_mode: str = "year",
    update_old_non_daily: bool = False,
) -> None:
    process_twse_data(
        cols=_coerce_cols(cols),
        storage_mode="cloud",
        batch_size=None,
        bucket_mode=bucket_mode,
        update_old_non_daily=update_old_non_daily,
    )


def run_local_clean(
    *,
    cols: Optional[Union[str, List[str]]] = None,
    batch_size: int = 500,
    bucket_mode: str = "year",
    update_old_non_daily: bool = False,
) -> None:
    process_twse_data(
        cols=_coerce_cols(cols),
        storage_mode="local",
        batch_size=batch_size,
        bucket_mode=bucket_mode,
        update_old_non_daily=update_old_non_daily,
    )


__all__ = [
    "run_cloud_staging_clean",
    "run_cloud_clean",
    "run_local_clean",
]

if __name__ == "__main__":

    # 清全部，雲端+staging、每批 500 檔、以 year 分桶
    run_cloud_staging_clean(update_old_non_daily=False)

    # 只清「三大法人買賣超日報」，用 month 分桶（且允許非日頻回補/覆寫舊資料）
    # run_cloud_staging_clean(cols="三大法人買賣超日報", bucket_mode="month", update_old_non_daily=True)

    # 在本機路徑做清理
    # run_local_clean(cols=["每日收盤行情"], batch_size=300, bucket_mode="year", update_old_non_daily=False)
