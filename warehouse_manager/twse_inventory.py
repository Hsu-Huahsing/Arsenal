# -*- coding: utf-8 -*-
"""
user_lab/warehouse_twse.py

用途：
    - 當成你日常開 Arsenal 時的「TWSE 主控台」。
    - 只做「前端介面」兩件事：
        1) 提供純函式介面 load_twse_dashboard()，回傳 dashboard dict，給任何 .py script 使用。
        2) 提供互動用介面 init_twse_dashboard()，幫你在本模組建立 tw_* 變數並印出總覽。

底層邏輯（掃描 source/cleaned、schema 判斷、孤兒檔統計、log/errorlog 攤平等）
全部集中在 warehouse_manager.twse_inventory.build_twse_dashboard()，
這裡不再重複實作，只負責呼叫與整理介面。
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
from typing import Dict, Any

import sys
import pandas as pd

# ---------------------------------------------------------------------------
# 0. sys.path 設定：讓 core_engine / warehouse_manager / StevenTricks 可以被 import
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
# 1. 匯入 TWSE 倉庫引擎 & stock_lab 工具
# ---------------------------------------------------------------------------

from warehouse_manager.twse_inventory import (
    build_twse_dashboard,
    prune_orphan_cleaned,  # 若未來要做實體刪除，可以直接從這裡呼叫
)

from core_engine.app.stock_lab import (
    quick_stock_vs_stock,
    quick_indicators_for_stock,
)

# ---------------------------------------------------------------------------
# 2. 純函式介面：load_twse_dashboard（給 script / .py 使用）
# ---------------------------------------------------------------------------

def load_twse_dashboard(
    today: date | None = None,
    include_log: bool = True,
    include_errorlog: bool = True,
) -> Dict[str, Any]:
    """
    純函式版介面：
        - 不動 globals()
        - 不印任何東西
    回傳一個 dict，key 跟 build_twse_dashboard 完全一致，例如：
        - "today"
        - "source_detail"
        - "cleaned_detail"
        - "item_summary"
        - "relation_status"
        - "missing"
        - "orphan"
        - "log_summary"
        - "error_log"
        - "overall_summary"
        - "source_item_summary"
        - "cleaned_item_summary"
        - "relation_counts"
        - "lag_item"
        - "lag_item_top"
        - "missing_top"
        - "orphan_top"
        - "error_log_flat"
        - "error_reason_summary"
        - "error_date_summary"
        - "error_item_summary"
        - "expected_item_master"
        - "expected_item_status"
        - "unexpected_items"
        - "schema_detail"
        - "schema_item_summary"
        等等。
    """
    dash = build_twse_dashboard(
        today=today,
        include_log=include_log,
        include_errorlog=include_errorlog,
    )
    return dash

# ---------------------------------------------------------------------------
# 3. 互動用 summary 列印小工具
# ---------------------------------------------------------------------------

def _print_twse_summary(dash: Dict[str, Any]) -> None:
    """
    把 build_twse_dashboard 回傳的 dict，印成一段簡短總覽。
    """
    overall = dash.get("overall_summary")
    if not isinstance(overall, pd.DataFrame) or overall.empty:
        print("[TWSE 倉庫總覽] 尚無資料")
        return

    row = overall.iloc[0]

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

        # 缺少 cleaned 的 item 清單
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

    # config.collection 之外的「意外 item」
    unexpected = dash.get("unexpected_items")
    if isinstance(unexpected, pd.DataFrame) and not unexpected.empty:
        print(
            f"  （另外有 {len(unexpected)} 個 item 出現在 source/cleaned，"
            f"但不在 config.collection，可看 tw_unexpected_items）"
        )

# ---------------------------------------------------------------------------
# 4. 互動版介面：init_twse_dashboard（給 IPython / Python console 用）
# ---------------------------------------------------------------------------

def init_twse_dashboard(
    today: date | None = None,
    include_log: bool = True,
    include_errorlog: bool = True,
    verbose: bool = True,
) -> None:
    """
    給「互動環境」用的版本：

        - 內部呼叫 load_twse_dashboard()
        - 把結果展開成 tw_* 變數丟到本模組的 globals()
        - 視 verbose 決定是否印 summary

    用法（互動環境）：
        >>> import user_lab.warehouse_twse as tw
        >>> tw.init_twse_dashboard()
        >>> tw.tw_overall_summary
        >>> tw.tw_orphan_pairs.head()
    """
    dash = load_twse_dashboard(
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
            tw_schema_detail=dash.get("schema_detail"),
            tw_schema_item_summary=dash.get("schema_item_summary"),
        )
    )

    if verbose:
        _print_twse_summary(dash)
        print("\n[提示] 你現在可以在互動環境裡直接操作這些變數，例如：")
        print("  tw_item_summary.head()")
        print("  tw_relation_status.query(\"relation == 'source_only'\").head()")
        print("  tw_orphan_pairs[['item', 'subitem', 'cleaned_path']].head()")

# ---------------------------------------------------------------------------
# 5. 說明：為什麼這裡不做 __main__ / argparse？
# ---------------------------------------------------------------------------
#
# 你前面已經明確說過：
#   - 「user_lab/warehouse_twse.py 是要可以讓我在 script 裡面直接使用的，
#      我不需要任何 ipython 或 terminal/cmd」
#
# 所以：
#   - CLI 報表（run_report, argparse 等）全部放在 warehouse_manager.twse_inventory 裡；
#   - user_lab 這支檔案只扮演「前端介面 + 互動主控台」的角色；
#   - 不寫 if __name__ == '__main__':，避免把 script 跑起來就被綁去印報表。
#
# 若未來你真的想在 command line 下直接跑 TWSE 報表：
#   → 請直接從 warehouse_manager.twse_inventory.import run_report，
#      另外寫一支專門的 CLI .py，會比較乾淨。
#
# ---------------------------------------------------------------------------
# 6. 附：若 script 想直接用 dict 介面，建議這樣寫
# ---------------------------------------------------------------------------
#
# from user_lab.warehouse_twse import load_twse_dashboard
#
# dash = load_twse_dashboard()
# tw_overall_summary = dash["overall_summary"]
# tw_orphan_pairs = dash["orphan"]
#
# # e.g.
# print(tw_overall_summary)
# print(tw_orphan_pairs[["item", "subitem", "cleaned_path"]].head())
#
# ---------------------------------------------------------------------------
