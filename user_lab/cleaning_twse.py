# -*- coding: utf-8 -*-
"""
user_lab/cleaning_twse.py

用途（使用者視角）：
    - 提供「一行就能叫用」的 TWSE 清理入口，不再出現任何 CLI / IPython 教學。
    - 真正的清理邏輯全部在 core_engine.data_cleaning.twse.process_twse_data 裡面實作。

設計原則：
    - 只做參數防呆與路徑可 import 的設定
    - 不包裝多餘邏輯，不在 user_lab 做資料處理
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# 0. sys.path 設定：讓 core_engine / StevenTricks 可以被 import
# ---------------------------------------------------------------------------

def _ensure_sys_path() -> Path:
    """
    確保 Arsenal 專案根目錄與 StevenTricks 目錄可被 import。
    回傳：PROJECT_ROOT（Arsenal 根目錄）
    """
    if "__file__" in globals():
        # .../Arsenal/user_lab/cleaning_twse.py -> parents[1] = .../Arsenal
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path.cwd()

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    steven_tricks_root = project_root.parent / "StevenTricks"
    if steven_tricks_root.exists() and str(steven_tricks_root) not in sys.path:
        sys.path.append(str(steven_tricks_root))

    return project_root


PROJECT_ROOT = _ensure_sys_path()


# ---------------------------------------------------------------------------
# 1. 匯入工具層主入口
# ---------------------------------------------------------------------------

try:
    from core_engine.data_cleaning.twse import process_twse_data
except Exception as e:
    raise ImportError(
        "無法匯入 core_engine.data_cleaning.twse.process_twse_data。\n"
        f"- PROJECT_ROOT={PROJECT_ROOT}\n"
        "- 請確認你是從 Arsenal 專案結構下執行，且 core_engine/ 目錄存在。\n"
        "- 若你把 Arsenal 移動位置，請確認 sys.path 已包含 Arsenal 根目錄。\n"
        f"- 原始錯誤：{type(e).__name__}: {e}"
    ) from e


# ---------------------------------------------------------------------------
# 2. 參數防呆
# ---------------------------------------------------------------------------

_ALLOWED_BUCKET_MODES = {"all", "year", "quarter", "month", "day"}
_ALLOWED_STORAGE_MODES = {"cloud", "cloud_staging", "local"}


def _coerce_cols(cols: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """
    防呆：
      - None -> None
      - "三大法人買賣超日報" -> ["三大法人買賣超日報"]
      - ["A","B"] -> ["A","B"]（同時 strip，移除空字串）
    """
    if cols is None:
        return None
    if isinstance(cols, str):
        s = cols.strip()
        return [s] if s else None
    out = [str(x).strip() for x in cols if str(x).strip()]
    return out if out else None


def _coerce_bucket_mode(bucket_mode: str) -> str:
    """
    bucket_mode 防呆：允許大小寫混用；不在允許清單就直接報錯。
    """
    bm = (bucket_mode or "all").strip().lower()
    if bm not in _ALLOWED_BUCKET_MODES:
        raise ValueError(
            f"bucket_mode 必須是 {sorted(_ALLOWED_BUCKET_MODES)} 之一，收到：{bucket_mode!r}"
        )
    return bm


def _coerce_batch_size(batch_size: Optional[int]) -> Optional[int]:
    """
    batch_size 防呆：
      - None -> None
      - <=0 直接報錯（你不會想要 batch_size=0 這種「看起來有跑但其實不處理」的行為）
    """
    if batch_size is None:
        return None
    if not isinstance(batch_size, int):
        raise TypeError(f"batch_size 必須是 int 或 None，收到：{type(batch_size).__name__}")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必須 > 0，收到：{batch_size}")
    return batch_size


def _run(
    *,
    storage_mode: str,
    cols: Optional[Union[str, List[str]]] = None,
    batch_size: Optional[int] = None,
    bucket_mode: str = "year",
    update_old_non_daily: bool = False,
) -> None:
    """
    統一入口：集中做防呆，避免三個 run_* 重複堆參數與漏改。
    """
    sm = (storage_mode or "cloud").strip().lower()
    if sm not in _ALLOWED_STORAGE_MODES:
        raise ValueError(
            f"storage_mode 必須是 {sorted(_ALLOWED_STORAGE_MODES)} 之一，收到：{storage_mode!r}"
        )

    process_twse_data(
        cols=_coerce_cols(cols),
        storage_mode=sm,
        batch_size=_coerce_batch_size(batch_size),
        bucket_mode=_coerce_bucket_mode(bucket_mode),
        update_old_non_daily=bool(update_old_non_daily),
    )


# ---------------------------------------------------------------------------
# 3. 對外暴露的三個簡單入口函式
# ---------------------------------------------------------------------------

def run_cloud_staging_clean(
    *,
    cols: Optional[Union[str, List[str]]] = None,
    batch_size: int = 500,
    bucket_mode: str = "year",
    update_old_non_daily: bool = False,
) -> None:
    """
    雲端 source + 本機 staging cleaned（分批同步回雲端 cleaned）
    """
    _run(
        storage_mode="cloud_staging",
        cols=cols,
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
    """
    純雲端：直接對雲端 cleaned 寫入（無 staging）
    """
    _run(
        storage_mode="cloud",
        cols=cols,
        batch_size=None,  # cloud 模式不使用 batch_size
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
    """
    完全本機：會清空 db_local_root 底下的 source/ cleaned/ log.pkl（依 twse.py 的 local 規則）
    """
    _run(
        storage_mode="local",
        cols=cols,
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
