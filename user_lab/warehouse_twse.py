# -*- coding: utf-8 -*-
"""
user_lab/warehouse_twse.py

用途：
    - 當成 Arsenal 的「TWSE 倉庫主控台」。
    - 調用 warehouse_manager.twse_inventory.build_twse_dashboard，
      幫你印出精緻版總覽，並回傳 dashboard dict。

使用情境：

1) 在 PyCharm console / 任何 .py：

    from user_lab.warehouse_twse import print_twse_summary

    dash = print_twse_summary()
    # 上面這行會印出總覽，同時把完整 dash dict 回傳
    dash["orphan_top"].head()
    dash["missing_top"].head()

2) 如果你喜歡 tw_* 變數（互動環境）：

    import user_lab.warehouse_twse as tw

    tw.init_twse_dashboard()
    tw.tw_overall_summary
    tw.tw_orphan_pairs.head()
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict

import pandas as pd

from warehouse_manager.twse_inventory import build_twse_dashboard


# ---------------------------------------------------------------------------
# 1. 純函式介面：不印字，只回傳 dict
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
    直接回傳 build_twse_dashboard 的結果 dict。
    """
    dash = build_twse_dashboard(
        today=today,
        include_log=include_log,
        include_errorlog=include_errorlog,
    )
    return dash


# ---------------------------------------------------------------------------
# 2. 共用的「精緻文字總覽」印法
# ---------------------------------------------------------------------------

def _print_summary(dash: Dict[str, Any]) -> None:
    """
    給 print_twse_summary / init_twse_dashboard 共用的印法。
    專心處理你要的「精緻版總覽」。
    """
    overall = dash.get("overall_summary")

    if not isinstance(overall, pd.DataFrame) or overall.empty:
        print("[TWSE 倉庫總覽] 尚無 overall_summary 資料（可能 cleaned / source 都是空的）")
        return

    row = overall.iloc[0]

    # 1) 整體檔案與 pair 統計
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

    # 2) 依 config.collection 檢查「設計上的 item 覆蓋狀況」
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

        # 哪些 item 還沒有 cleaned
        missing_cleaned_items = exp_status.loc[
            exp_status["missing_cleaned"], "item"
        ].tolist()
        if missing_cleaned_items:
            preview = "、".join(missing_cleaned_items[:5])
            more = (
                ""
                if len(missing_cleaned_items) <= 5
                else f"... 等共 {len(missing_cleaned_items)} 項"
            )
            print(f"  缺少 cleaned 的 item：{preview}{more}")

    # 3) 不在 config.collection 裡的 item
    unexpected = dash.get("unexpected_items")
    if isinstance(unexpected, pd.DataFrame) and not unexpected.empty:
        print(
            f"  （另外有 {len(unexpected)} 個 item 出現在 source/cleaned，"
            f"但不在 config.collection，可看 dash['unexpected_items']）"
        )

    # 4) 想要的話可以再加「落後最嚴重 / 缺漏最多」Top N
    lag_item_top = dash.get("lag_item_top")
    missing_top = dash.get("missing_top")
    orphan_top = dash.get("orphan_top")

    if isinstance(lag_item_top, pd.DataFrame) and not lag_item_top.empty:
        print("\n  ▶ 落後天數最高的 item（前幾名）：")
        cols = [c for c in ["item", "max_days_lag", "n_files"] if c in lag_item_top.columns]
        print("    " + lag_item_top[cols].head(5).to_string(index=False).replace("\n", "\n    "))

    if isinstance(missing_top, pd.DataFrame) and not missing_top.empty:
        print("\n  ▶ 缺少 cleaned 的 (item, subitem) 範例（前幾筆）：")
        cols = [c for c in ["item", "subitem", "source_mtime", "source_status"] if c in missing_top.columns]
        print("    " + missing_top[cols].head(5).to_string(index=False).replace("\n", "\n    "))

    if isinstance(orphan_top, pd.DataFrame) and not orphan_top.empty:
        print("\n  ▶ 孤兒（只有 cleaned）(item, subitem) 範例（前幾筆）：")
        cols = [c for c in ["item", "subitem", "cleaned_mtime", "cleaned_status"] if c in orphan_top.columns]
        print("    " + orphan_top[cols].head(5).to_string(index=False).replace("\n", "\n    "))


# ---------------------------------------------------------------------------
# 3. 對外主函式：印出精緻總覽 + 回傳 dash
# ---------------------------------------------------------------------------

def print_twse_summary(
    today: date | None = None,
    include_log: bool = True,
    include_errorlog: bool = True,
) -> Dict[str, Any]:
    """
    主入口（推薦用法）：

        from user_lab.warehouse_twse import print_twse_summary
        dash = print_twse_summary()

    會：
        1) 呼叫 build_twse_dashboard()
        2) 印出一次精緻總覽（你熟悉的那種格式）
        3) 把完整 dash dict 回傳，方便後續 DataFrame 分析。
    """
    dash = load_twse_dashboard(
        today=today,
        include_log=include_log,
        include_errorlog=include_errorlog,
    )
    _print_summary(dash)
    return dash


# ---------------------------------------------------------------------------
# 4. tw_* 風格的互動版：把結果塞進本模組 globals()
# ---------------------------------------------------------------------------

def init_twse_dashboard(
    today: date | None = None,
    include_log: bool = True,
    include_errorlog: bool = True,
    verbose: bool = True,
) -> None:
    """
    給互動環境（或你想用 tw_* 變數）用的版本：

        import user_lab.warehouse_twse as tw
        tw.init_twse_dashboard()
        tw.tw_overall_summary
        tw.tw_orphan_pairs.head()

    會：
        - 呼叫 load_twse_dashboard()
        - 把常用的 DataFrame 展開成 tw_xxx 變數放在本模組 globals()
        - 若 verbose=True，順便印一次精緻總覽。
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
        _print_summary(dash)


# ---------------------------------------------------------------------------
# 5. 直接當 script 執行時的入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 直接：
    #   python -m user_lab.warehouse_twse
    # 或
    #   python user_lab/warehouse_twse.py
    # 會印一份總覽。
    print_twse_summary()
