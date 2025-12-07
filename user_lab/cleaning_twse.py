# -*- coding: utf-8 -*-
"""
user_lab/cleaning_twse.py

用途（使用者視角）：
    - 提供「一行就能叫用」的 TWSE cleaned 清理入口，不再出現任何 CLI / IPython 教學。
    - 真正的清理邏輯全部在工具層 core_engine.data_cleaning.twse 裡面實作。

使用方式（在 PyCharm / IPython Console 等互動環境）：
    from user_lab.cleaning_twse import run_cloud_staging_clean

    # 典型用法：雲端 + staging，本機處理後再同步回雲端
    run_cloud_staging_clean(batch_size=10, bucket_mode="year")

    # 或者只想在雲端直接清理（不用 staging）
    # from user_lab.cleaning_twse import run_cloud_clean
    # run_cloud_clean(bucket_mode="year")

    # 或者完全只用本機 TWSE DB（不碰雲端）
    # from user_lab.cleaning_twse import run_local_clean
    # run_local_clean(batch_size=500, bucket_mode="year")

說明：
    - 這個檔案的責任只有兩件事：
        1) 處理 sys.path，讓 core_engine 可以被順利 import。
        2) 封裝工具層的 process_twse_data，提供乾淨的入口函式。
    - 所有「真正的清理流程、DB / staging / bucket_mode 邏輯」都在
      core_engine.data_cleaning.twse.process_twse_data 內部實作。
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Optional

# ---------------------------------------------------------------------------
# 0. sys.path 設定：讓 core_engine 可以被 import
#    （符合最高規則：深層 library 不改 sys.path，交給 user_lab 處理）
# ---------------------------------------------------------------------------

if "__file__" in globals():
    # 正常以 .py 檔案形式執行時
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Arsenal
else:
    # 例如在某些互動環境直接 %run 或貼 code 時
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 若 StevenTricks 在同一層專案目錄（和 Arsenal 並列），一併加入
STEVEN_TRICKS_ROOT = PROJECT_ROOT.parent / "StevenTricks"
if STEVEN_TRICKS_ROOT.exists() and str(STEVEN_TRICKS_ROOT) not in sys.path:
    sys.path.append(str(STEVEN_TRICKS_ROOT))

# ---------------------------------------------------------------------------
# 1. 匯入工具層主入口
# ---------------------------------------------------------------------------

from core_engine.data_cleaning.twse import process_twse_data


# ---------------------------------------------------------------------------
# 2. 對外暴露的三個簡單入口函式
# ---------------------------------------------------------------------------

def run_cloud_staging_clean(
    *,
    cols: Optional[List[str]] = None,
    batch_size: int = 500,
    bucket_mode: str = "year",
) -> None:
    """
    雲端 + 本機暫存（staging）模式：

    行為概要：
        - source / cleaned 主體仍在雲端 TWSE DB。
        - 每一輪把 cleaned 拉到本機 staging 目錄，跑完清理後再同步回雲端。
        - 適合大批量更新、避免 iCloud / 網路不穩導致中途壞檔。

    參數：
        cols:
            限制只清某幾個 TWSE item（資料夾名 / collection key），例如：
                cols=["三大法人買賣超日報", "信用交易統計"]
            預設 None 代表「清全部可見 item」。
        batch_size:
            cloud_staging 模式下，每一輪最多處理幾個檔案。
            例如 500：一次 staging 500 檔，處理完再進下一輪。
        bucket_mode:
            日期分桶模式，由工具層解釋：
                "all"     → 一張表吃全部日期
                "year"    → 每年一張表
                "quarter" → 每季一張表
                "month"   → 每月一張表
                "day"     → 每日一張表
    """
    process_twse_data(
        cols=cols,
        storage_mode="cloud_staging",
        batch_size=batch_size,
        bucket_mode=bucket_mode,
    )


def run_cloud_clean(
    *,
    cols: Optional[List[str]] = None,
    bucket_mode: str = "year",
) -> None:
    """
    完全雲端模式（沒有 staging）：

    行為概要：
        - 直接在雲端 TWSE DB 路徑上清理，不拉到本機暫存。
        - 適合檔案數量不多、或你確認 iCloud / NAS 很穩定時使用。

    參數：
        cols:
            同 run_cloud_staging_clean。
        bucket_mode:
            同 run_cloud_staging_clean。
    """
    process_twse_data(
        cols=cols,
        storage_mode="cloud",
        batch_size=None,  # cloud 模式內部不會用到 batch_size
        bucket_mode=bucket_mode,
    )


def run_local_clean(
    *,
    cols: Optional[List[str]] = None,
    batch_size: int = 500,
    bucket_mode: str = "year",
) -> None:
    """
    完全本機模式：

    行為概要：
        - source / cleaned 都走 config.paths 裡設定的 db_local_root。
        - 不觸及任何雲端路徑，適合離線開發 / 重構 schema / 做大規模實驗。

    參數：
        cols:
            同 run_cloud_staging_clean。
        batch_size:
            在 local 模式下，每輪最多處理幾檔。
        bucket_mode:
            同 run_cloud_staging_clean。
    """
    process_twse_data(
        cols=cols,
        storage_mode="local",
        batch_size=batch_size,
        bucket_mode=bucket_mode,
    )


# ---------------------------------------------------------------------------
# 3. 明確限定本模組對外提供的 API
# ---------------------------------------------------------------------------

__all__ = [
    "run_cloud_staging_clean",
    "run_cloud_clean",
    "run_local_clean",
]

if __name__ == "__main__" :

    # 清全部，雲端+staging、每批 500 檔、以 year 分桶
    run_cloud_staging_clean()

    # 只清「三大法人買賣超日報」，用 month 分桶
    # run_cloud_staging_clean(cols="三大法人買賣超日報", bucket_mode="month")

    # 在本機路徑做清理
    # run_local_clean(cols=["每日收盤行情"], batch_size=300, bucket_mode="year")