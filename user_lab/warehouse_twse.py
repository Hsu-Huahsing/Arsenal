# -*- coding: utf-8 -*-
"""
user_lab/warehouse_twse.py

用途：
    - 當成 Arsenal 的「TWSE 倉庫主控台」。
    - 調用 warehouse_manager.twse_inventory.build_twse_dashboard，
      幫你印出精緻版總覽，並（選擇性）把結果展開成 tw_* 變數。

使用情境：

1) 在任何 .py（推薦）：
    from user_lab.warehouse_twse import print_twse_summary

    dash = print_twse_summary()
    dash["overall_summary"]
    dash["orphan_top"].head()

2) 在互動環境想要 tw_* 變數：
    import user_lab.warehouse_twse as tw

    tw.init_twse_dashboard()
    tw.tw_overall_summary
    tw.tw_orphan_pairs[["item", "subitem", "cleaned_path"]].head()
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
    專心處理「精緻版總覽」。
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

    # 4) Top N：誰最落後、誰缺最多、誰是孤兒
    lag_item_top = dash.get("lag_item_top")
    missing_top = dash.get("missing_top")
    orphan_top = dash.get("orphan_top")

    if isinstance(lag_item_top, pd.DataFrame) and not lag_item_top.empty:
        print("\n  ▶ 落後天數最高的 item（前幾名）：")
        cols = [c for c in ["item", "max_days_lag", "n_files"] if c in lag_item_top.columns]
        txt = lag_item_top[cols].head(5).to_string(index=False)
        print("    " + txt.replace("\n", "\n    "))

    if isinstance(missing_top, pd.DataFrame) and not missing_top.empty:
        print("\n  ▶ 缺少 cleaned 的 (item, subitem) 範例（前幾筆）：")
        cols = [c for c in ["item", "subitem", "source_mtime", "source_status"] if c in missing_top.columns]
        txt = missing_top[cols].head(5).to_string(index=False)
        print("    " + txt.replace("\n", "\n    "))

    if isinstance(orphan_top, pd.DataFrame) and not orphan_top.empty:
        print("\n  ▶ 孤兒（只有 cleaned）(item, subitem) 範例（前幾筆）：")
        cols = [c for c in ["item", "subitem", "cleaned_mtime", "cleaned_status"] if c in orphan_top.columns]
        txt = orphan_top[cols].head(5).to_string(index=False)
        print("    " + txt.replace("\n", "\n    "))


# ---------------------------------------------------------------------------
# 3. 對外主函式：印出精緻總覽 + 回傳 dash
# ---------------------------------------------------------------------------
def _print_summary(dash: Dict[str, Any]) -> None:
    """
    依照指定格式，印出 TWSE 倉庫總覽 + Crawler 成效 + 落後 item + error_log 摘要。
    """
    overall = dash.get("overall_summary")
    if not isinstance(overall, pd.DataFrame) or overall.empty:
        print("[TWSE 倉庫總覽] 尚無 overall_summary 資料（可能 cleaned / source 都是空的）")
        return

    row = overall.iloc[0]
    as_of = row.get("as_of")

    expected_status = dash.get("expected_item_status")
    cleaned_item_summary = dash.get("cleaned_item_summary")
    schema_item_summary = dash.get("schema_item_summary")
    cleaned_detail = dash.get("cleaned_detail")
    orphan_true = dash.get("orphan")  # 真孤兒
    relation_counts = dash.get("relation_counts")
    lag_item_top = dash.get("lag_item_top")

    error_flat = dash.get("error_log_flat")
    error_item_summary = dash.get("error_item_summary")

    # ------------------------
    # [TWSE 倉庫總覽]
    # ------------------------
    print(f"[TWSE 倉庫總覽] 截至 {as_of}")

    # 1) collection item 覆蓋狀況
    if isinstance(expected_status, pd.DataFrame) and not expected_status.empty:
        n_expected = int(expected_status.shape[0])

        n_src_ok = int(expected_status["has_source"].sum()) if "has_source" in expected_status.columns else 0
        n_src_missing = int(expected_status["missing_source"].sum()) if "missing_source" in expected_status.columns else max(n_expected - n_src_ok, 0)

        n_cln_ok = int(expected_status["has_cleaned"].sum()) if "has_cleaned" in expected_status.columns else 0
        n_cln_missing = int(expected_status["missing_cleaned"].sum()) if "missing_cleaned" in expected_status.columns else max(n_expected - n_cln_ok, 0)

        n_schema_ok = int(expected_status["has_schema"].sum()) if "has_schema" in expected_status.columns else 0
        n_schema_missing = int(expected_status["missing_schema"].sum()) if "missing_schema" in expected_status.columns else max(n_expected - n_schema_ok, 0)

        print(f"TWSE 設定 collection item 數：{n_expected}；")
        print(f"有 source 的：{n_src_ok}（缺少 {n_src_missing}）")
        print(f"有 cleaned 的：{n_cln_ok}（缺少 {n_cln_missing}）")
        print(f"有 schema 的：{n_schema_ok}（缺少 {n_schema_missing}）")
    else:
        print("（找不到 expected_item_status，可先確認 config.collection 設定）")

    # 2) cleaned item 與 schema 的一對一匹配狀況（實際 cleaned 為基準）
    print("每個 cleaned item 都會匹配一個 schema")

    if isinstance(cleaned_item_summary, pd.DataFrame) and not cleaned_item_summary.empty:
        tmp = cleaned_item_summary.copy()
        if "has_schema" in tmp.columns:
            matched_mask = tmp["has_schema"] == True
        else:
            matched_mask = pd.Series(False, index=tmp.index)

        n_cleaned_items = int(tmp.shape[0])
        n_matched = int(matched_mask.sum())
        unmatched_items = tmp.loc[~matched_mask, "item"].tolist()
        n_unmatched = len(unmatched_items)

        print(f"目前有達成匹配的有 {n_matched}")
        if n_unmatched > 0:
            preview = "、".join(unmatched_items[:5])
            more = "" if n_unmatched <= 5 else f"... 等共 {n_unmatched} 項"
            print(f"沒有達成匹配的有 {n_unmatched}（{preview}{more}）")
        else:
            print("沒有達成匹配的有 0（所有 cleaned item 均已配置 schema）")
    else:
        print("目前沒有任何 cleaned item，無法檢查 schema 匹配。")

    # ------------------------
    # [TWSE Crawler成效]
    # ------------------------
    print("\n[TWSE Crawler成效] ")

    n_items_cleaned = int(row.get("n_items_cleaned", 0) or 0)
    n_pairs_total = int(row.get("n_pairs_total", 0) or 0)
    n_pairs_source = int(row.get("n_pairs_source", 0) or 0)
    n_pairs_source_only = int(row.get("n_pairs_source_only", 0) or 0)

    print(f"已抓取 item 數（cleaned）：{n_items_cleaned}；")
    print(f"預計要抓取的 source 檔案數：{n_pairs_total}；")
    print(f"已經抓取的 source 檔案數：{n_pairs_source}；")
    print(f"目前等待被 cleaned 的檔案數：{n_pairs_source_only}；")

    # cleaned 檔案數與 schema 檔案數、匹配狀況
    if isinstance(cleaned_detail, pd.DataFrame) and not cleaned_detail.empty:
        n_cleaned_files = int(cleaned_detail.shape[0])
        if "file_role" in cleaned_detail.columns:
            n_schema_files = int((cleaned_detail["file_role"] == "schema").sum())
        else:
            n_schema_files = 0

        # 以 item 為單位，看有哪些 item 同時出現在 cleaned 與 schema
        if isinstance(schema_item_summary, pd.DataFrame) and not schema_item_summary.empty:
            cleaned_items = set(cleaned_item_summary["item"].unique()) if isinstance(cleaned_item_summary, pd.DataFrame) and not cleaned_item_summary.empty else set()
            schema_items = set(schema_item_summary["item"].unique())
            items_with_both = cleaned_items & schema_items
            items_without_schema = cleaned_items - schema_items
            n_item_with_both = len(items_with_both)
            n_item_without_schema = len(items_without_schema)
        else:
            n_item_with_both = 0
            n_item_without_schema = 0

        print(
            f"    其中 cleaned 完成的檔案數為 {n_cleaned_files} "
            f"(其中有 {n_schema_files} 個是 schema 檔案；"
            f"{n_item_with_both} 個 item 的 cleaned 與 schema 有成功匹配，"
            f"{n_item_without_schema} 個 item 仍缺 schema)"
        )
    else:
        print("    目前尚無 cleaned 檔案。")

    # 真孤兒（整個 item 在 source 沒任何檔案）
    if isinstance(orphan_true, pd.DataFrame) and not orphan_true.empty:
        n_true_orphan = int(orphan_true.shape[0])
    else:
        n_true_orphan = 0

    print(
        f"    另外，發現只有 cleaned 後的檔案，卻沒有對應的 source 檔案，"
        f"共 {n_true_orphan} 筆（真孤兒）。"
    )

    # ------------------------
    # ▶ 落後天數最高的 item
    # ------------------------
    if isinstance(lag_item_top, pd.DataFrame) and not lag_item_top.empty:
        print("  ▶ 落後天數最高的 item：")

        # 把 lag_item_top 跟 relation_counts 連起來，拿到每個 item 的 source_only 檔案數（尚待 cleaned）
        if isinstance(relation_counts, pd.DataFrame) and not relation_counts.empty:
            rc = relation_counts.copy()
            if "source_only" not in rc.columns:
                rc["source_only"] = 0
            merged = lag_item_top.merge(rc[["item", "source_only"]], on="item", how="left")
        else:
            merged = lag_item_top.copy()
            merged["source_only"] = 0

        for _, r in merged.head(5).iterrows():
            item_name = r.get("item")
            last_date = r.get("max_mtime")
            max_lag = int(r.get("max_days_lag", 0) or 0)
            to_clean = int(r.get("source_only", 0) or 0)

            last_date_str = str(last_date) if pd.notna(last_date) else "未知日期"
            print(
                f"           {item_name} （最新一次檔案記錄為 {last_date_str} ，"
                f"至今共落後 {max_lag} 天，還要補齊 {to_clean} 個檔案）"
            )
    else:
        print("  ▶ 落後天數最高的 item：目前沒有 cleaned 檔案，無法排序。")

    # ------------------------
    # ▶ error_log 統計
    # ------------------------
    if isinstance(error_flat, pd.DataFrame) and not error_flat.empty and \
       isinstance(error_item_summary, pd.DataFrame) and not error_item_summary.empty:

        n_items_in_error = int(error_item_summary.shape[0])
        n_errors_total = int(error_flat.shape[0])

        print(
            f"  ▶ 位於 error_log 的 item 共 {n_items_in_error} 筆，"
            f"需要重新 crawler 的檔案數共 {n_errors_total} 筆："
        )

        # 依 item 排序，挑前幾個代表性 item
        for _, r in error_item_summary.head(4).iterrows():
            item_name = r.get("item")
            n_errors_item = int(r.get("n_errors", 0) or 0)

            sub = error_flat[error_flat["item"] == item_name].copy()

            # 日期範圍：優先用 log_date，其次用 date
            if "log_date" in sub.columns and sub["log_date"].notna().any():
                dmin = sub["log_date"].min()
                dmax = sub["log_date"].max()
            elif "date" in sub.columns and sub["date"].notna().any():
                dmin = sub["date"].min()
                dmax = sub["date"].max()
            else:
                dmin = dmax = None

            if dmin is not None and pd.notna(dmin):
                dmin_str = str(dmin)
            else:
                dmin_str = "未知日期"

            if dmax is not None and pd.notna(dmax):
                dmax_str = str(dmax)
            else:
                dmax_str = "未知日期"

            # 主要 error 原因：挑前幾個 error_reason
            if "error_reason" in sub.columns:
                reason_counts = (
                    sub["error_reason"]
                    .fillna("unknown")
                    .value_counts()
                )
                top_reasons = list(reason_counts.index[:3])
                reason_str = "、".join(top_reasons)
            else:
                reason_str = "unknown"

            print(
                f"         {item_name} 主要 error 記錄分布於 {dmin_str} ~ {dmax_str} "
                f"共 {n_errors_item} 筆，error 原因為 {reason_str} ..."
            )
    else:
        print("  ▶ 位於 error_log 的 item：目前沒有 errorlog.pkl 或內容為空。")


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
    from user_lab.warehouse_twse import print_twse_summary, init_twse_dashboard

    # 印一份總覽 + 拿到 dash dict
    dash = print_twse_summary()

    # 如果你比較習慣 tw_* 變數：
    import user_lab.warehouse_twse as tw

    tw.init_twse_dashboard()
    tw.tw_overall_summary
    tw.tw_orphan_pairs.head()  # 真孤兒
    tw.tw_status_dict["orphan_candidate"].head()  # 所有 cleaned_only，含 has_source_for_item
