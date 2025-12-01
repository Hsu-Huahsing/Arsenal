# -*- coding: utf-8 -*-
"""
user_lab/warehouse_twse.py

用途：
    - 當成你日常開 Arsenal 時的「主控台」。
    - 一次把：
        1) TWSE 倉庫現況（source / cleaned / 缺漏 / 孤兒 / log）
        2) stock_lab 的常用分析結果
      丟到互動環境裡，讓你自己點 DataFrame 看。

使用方式（在 IPython / Jupyter）：
    %run -i path/to/Arsenal/user_lab/warehouse_twse.py

執行後，你會在變數列表看到：
    tw_today
    tw_source_detail
    tw_cleaned_detail
    tw_item_summary
    tw_relation_status
    tw_missing_pairs
    tw_orphan_pairs
    tw_log_summary
    tw_error_log
    df_two_example
    df_multi_example
    等等
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# 0. sys.path 設定：讓 core_engine / warehouse_manager 可以被 import
# ---------------------------------------------------------------------------

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Arsenal
else:
    # 假設你在 Arsenal 目錄底下直接執行
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

ST_ROOT = PROJECT_ROOT.parent / "StevenTricks"
if ST_ROOT.exists() and str(ST_ROOT) not in sys.path:
    sys.path.append(str(ST_ROOT))

# ---------------------------------------------------------------------------
# 1. 匯入倉庫引擎 & stock_lab 工具
# ---------------------------------------------------------------------------

from warehouse_manager.twse_inventory import build_twse_dashboard

from core_engine.app.stock_lab import (
    quick_stock_vs_stock,
    quick_indicators_for_stock,
)

# ---------------------------------------------------------------------------
# 2. TWSE 倉庫 dashboard：只做「算 df & 丟到 global」，不在這裡做太多 print
# ---------------------------------------------------------------------------
def init_twse_dashboard(
    today: date | None = None,
    include_log: bool = True,
    include_errorlog: bool = True,
    verbose: bool = True,
) -> None:
    """
    把 TWSE 倉庫重要資訊整理成一組全域變數（全部都從 build_twse_dashboard 拿）：
        tw_today
        tw_status_dict
        tw_source_detail
        tw_cleaned_detail
        tw_item_summary
        tw_relation_status
        tw_missing_pairs
        tw_orphan_pairs
        tw_log_summary
        tw_error_log

        tw_overall_summary
        tw_source_item_summary
        tw_cleaned_item_summary
        tw_relation_counts
        tw_lag_item
        tw_lag_item_top
        tw_missing_top
        tw_orphan_top
        tw_error_log_flat
        tw_error_reason_summary
        tw_error_date_summary
        tw_error_item_summary
    """
    dash = build_twse_dashboard(
        today=today,
        include_log=include_log,
        include_errorlog=include_errorlog,
    )
    globals().update(
        dict(
            tw_today=dash["today"],
            tw_status_dict=dash,
            tw_source_detail=dash["source_detail"],
            tw_cleaned_detail=dash["cleaned_detail"],
            tw_item_summary=dash["item_summary"],
            tw_relation_status=dash["relation_status"],
            tw_missing_pairs=dash["missing"],
            tw_orphan_pairs=dash["orphan"],
            tw_log_summary=dash["log_summary"],
            tw_error_log=dash["error_log"],
            tw_overall_summary=dash["overall_summary"],
            tw_source_item_summary=dash["source_item_summary"],
            tw_cleaned_item_summary=dash["cleaned_item_summary"],
            tw_relation_counts=dash["relation_counts"],
            tw_lag_item=dash["lag_item"],
            tw_lag_item_top=dash["lag_item_top"],
            tw_missing_top=dash["missing_top"],
            tw_orphan_top=dash["orphan_top"],
            tw_error_log_flat=dash["error_log_flat"],
            tw_error_reason_summary=dash["error_reason_summary"],
            tw_error_date_summary=dash["error_date_summary"],
            tw_error_item_summary=dash["error_item_summary"],
            tw_expected_item_master=dash["expected_item_master"],
            tw_expected_item_status=dash["expected_item_status"],
            tw_unexpected_items=dash["unexpected_items"],
        )
    )
    if verbose:
        row = dash["overall_summary"].iloc[0]
        print(f"[TWSE 倉庫總覽] 截至 {row['as_of']}")
        print(
            f"  item 數（cleaned）：{int(row['n_items_cleaned'])}；"
            f"source 檔數：{int(row['n_source_files'])}；"
            f"cleaned 檔數：{int(row['n_cleaned_files'])}"
        )
        print(
            f"  (item, subitem) 組合：總 {int(row['n_pairs_total'])}；"
            f"both={int(row['n_pairs_both'])} / "
            f"source_only={int(row['n_pairs_source_only'])} / "
            f"cleaned_only={int(row['n_pairs_cleaned_only'])}"
        )
        print(
            f"  缺少 cleaned 的 pair：{int(row['n_missing_pairs'])}；"
            f"孤兒（只有 cleaned）pair：{int(row['n_orphan_pairs'])}"
        )

        # 依 config.collection 檢查「應有 item」是否有抓到 / 有清理 / 有 schema
        exp_status = dash.get("expected_item_status")
        if isinstance(exp_status, pd.DataFrame) and not exp_status.empty:
            n_expected = int(exp_status.shape[0])
            n_src_ok = int(exp_status["has_source"].sum())
            n_cln_ok = int(exp_status["has_cleaned"].sum())
            n_src_missing = int(exp_status["missing_source"].sum())
            n_cln_missing = int(exp_status["missing_cleaned"].sum())

            print(
                f"  TWSE 設定 collection item 數：{n_expected}；"
                f"其中有 source 的：{n_src_ok}（缺少 {n_src_missing}）"
            )
            print(
                f"                               "
                f"有 cleaned 的：{n_cln_ok}（缺少 {n_cln_missing}）"
            )

            # schema 覆蓋率
            if "has_schema" in exp_status.columns and "missing_schema" in exp_status.columns:
                n_schema_ok = int(exp_status["has_schema"].sum())
                n_schema_missing = int(exp_status["missing_schema"].sum())
                print(
                    f"                               "
                    f"有 schema 的：{n_schema_ok}（缺少 {n_schema_missing}）"
                )

                missing_schema_items = exp_status.loc[
                    exp_status["missing_schema"], "item"
                ].tolist()
                if missing_schema_items:
                    preview = "、".join(missing_schema_items[:5])
                    more = (
                        ""
                        if len(missing_schema_items) <= 5
                        else f"... 等共 {len(missing_schema_items)} 項"
                    )
                    print(f"  缺少 schema 的 item：{preview}{more}")

            missing_items = exp_status.loc[
                exp_status["missing_cleaned"], "item"
            ].tolist()
            if missing_items:
                preview = "、".join(missing_items[:5])
                more = (
                    ""
                    if len(missing_items) <= 5
                    else f"... 等共 {len(missing_items)} 項"
                )
                print(f"  缺少 cleaned 的 item：{preview}{more}")

        unexpected = dash.get("unexpected_items")
        if isinstance(unexpected, pd.DataFrame) and not unexpected.empty:
            print(
                f"  （另外有 {len(unexpected)} 個 item 出現在 source/cleaned，"
                f"但不在 config.collection，可看 tw_unexpected_items）"
            )


# ---------------------------------------------------------------------------
# 3. 主入口：你平常只要跑這支 .py 就好
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. 先把 TWSE 倉庫狀態載進來
    init_twse_dashboard()

    #
    # print("\n[提示] 你現在可以在互動環境裡直接操作這些變數，例如：")
    # print("  tw_item_summary.head()")
    # print("  tw_relation_status.query(\"relation == 'source_only'\").head()")
    # print("  df_two_example.tail()")
    # print("  df_multi_example.tail()")
