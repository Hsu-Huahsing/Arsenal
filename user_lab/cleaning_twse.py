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

# ---------------------------------------------------------------------------
# 0. sys.path 設定：讓 core_engine 可以被 import
# ---------------------------------------------------------------------------

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Arsenal
else:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

STEVEN_TRICKS_ROOT = PROJECT_ROOT.parent / "StevenTricks"
if STEVEN_TRICKS_ROOT.exists() and str(STEVEN_TRICKS_ROOT) not in sys.path:
    sys.path.append(str(STEVEN_TRICKS_ROOT))

# ---------------------------------------------------------------------------
# 1. 匯入工具層主入口
# ---------------------------------------------------------------------------

from core_engine.data_cleaning.twse import process_twse_data


def _coerce_cols(cols: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """
    防呆：
      - None -> None
      - "三大法人買賣超日報" -> ["三大法人買賣超日報"]
      - ["A","B"] -> ["A","B"]
    """
    if cols is None:
        return None
    if isinstance(cols, str):
        s = cols.strip()
        return [s] if s else None
    # 其他 iterable 就當成 list
    out = [str(x).strip() for x in cols if str(x).strip()]
    return out if out else None


# ---------------------------------------------------------------------------
# 2. 對外暴露的三個簡單入口函式
# ---------------------------------------------------------------------------

def run_cloud_staging_clean(
    *,
    cols: Optional[Union[str, List[str]]] = None,
    batch_size: int = 500,
    bucket_mode: str = "year",
) -> None:
    process_twse_data(
        cols=_coerce_cols(cols),
        storage_mode="cloud_staging",
        batch_size=batch_size,
        bucket_mode=bucket_mode,
    )


def run_cloud_clean(
    *,
    cols: Optional[Union[str, List[str]]] = None,
    bucket_mode: str = "year",
) -> None:
    process_twse_data(
        cols=_coerce_cols(cols),
        storage_mode="cloud",
        batch_size=None,  # cloud 模式內部不會用到 batch_size
        bucket_mode=bucket_mode,
    )


def run_local_clean(
    *,
    cols: Optional[Union[str, List[str]]] = None,
    batch_size: int = 500,
    bucket_mode: str = "year",
) -> None:
    process_twse_data(
        cols=_coerce_cols(cols),
        storage_mode="local",
        batch_size=batch_size,
        bucket_mode=bucket_mode,
    )


__all__ = [
    "run_cloud_staging_clean",
    "run_cloud_clean",
    "run_local_clean",
]

if __name__ == "__main__":

    # 清全部，雲端+staging、每批 500 檔、以 year 分桶
    run_cloud_staging_clean()

    # 只清「三大法人買賣超日報」，用 month 分桶
    # run_cloud_staging_clean(cols="三大法人買賣超日報", bucket_mode="month")

    # 在本機路徑做清理
    # run_local_clean(cols=["每日收盤行情"], batch_size=300, bucket_mode="year")
