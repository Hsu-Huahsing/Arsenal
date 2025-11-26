"""
TWSE 倉庫管理員
==============

提供幾個層級的檢查：

1. 檔案層級 inventory（source / cleaned）
   - 每個 item / subitem 的檔案大小、最後更新時間、落後天數、狀態（OK / LAGGING / STALE）。

2. source vs cleaned 對比
   - 哪些 (item, subitem) 只有 source、有 cleaned、兩邊都有。
   - 兩邊 mtime 差距多久（方便判斷「source 已更新但 cleaned 尚未重跑」）。

3. item 層級 summary
   - 每個 item 有幾個 subitem、最舊 / 最新 mtime、多少 OK / LAGGING / STALE。

4. 清理 log / error log 彙總
   - 若 schema_utils.LogMaintainer / errorlog.pkl 有正確設定，可載入作輔助判讀。
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
from typing import Optional, Tuple, Dict, Any

import pandas as pd

from config.paths import (
    dbpath_cleaned,
    dbpath_source,
    dbpath_log,
    dbpath_errorlog,
)
from schema_utils import (
    scan_pkl_tree,
    add_days_lag,
    add_status_by_lag,
    LogMaintainer,
)
from StevenTricks.file_utils import pickleio


# ---------------------------------------------------------------------------
# 1. 檔案層級掃描
# ---------------------------------------------------------------------------

def scan_cleaned_files(today: Optional[date] = None) -> pd.DataFrame:
    """
    掃描 cleaned DB 資料夾，列出所有 item / subitem 對應的檔案、大小與最後更新時間，
    並加上 days_lag / status。

    回傳欄位：
        - layer="cleaned"
        - item
        - subitem
        - path
        - size_mb
        - mtime
        - days_lag
        - status  (OK / LAGGING / STALE)
    """
    df = scan_pkl_tree(Path(dbpath_cleaned), layer="cleaned")
    if df.empty:
        return df
    df = add_days_lag(df, date_col="mtime", today=today)
    df = add_status_by_lag(df, lag_col="days_lag")
    return df


def scan_source_files(today: Optional[date] = None) -> pd.DataFrame:
    """
    掃描 source DB 資料夾（爬蟲原始檔），欄位同 scan_cleaned_files。
    """
    df = scan_pkl_tree(Path(dbpath_source), layer="source")
    if df.empty:
        return df
    df = add_days_lag(df, date_col="mtime", today=today)
    df = add_status_by_lag(df, lag_col="days_lag")
    return df


# ---------------------------------------------------------------------------
# 2. source vs cleaned 對比
# ---------------------------------------------------------------------------

def merge_source_cleaned(today: Optional[date] = None) -> pd.DataFrame:
    """
    以 (item, subitem) 為主鍵，把 source / cleaned 的檔案層級資訊做 outer join。

    主要欄位：
        - item, subitem
        - source_path, source_mtime, source_size_mb, source_days_lag, source_status
        - cleaned_path, cleaned_mtime, cleaned_size_mb, cleaned_days_lag, cleaned_status
        - relation : 'both' / 'source_only' / 'cleaned_only'
        - mtime_diff_days : cleaned_mtime - source_mtime（正值代表 cleaned 比 source 新）
    """
    src = scan_source_files(today=today)
    cln = scan_cleaned_files(today=today)

    src_cols = {
        "path": "source_path",
        "size_mb": "source_size_mb",
        "mtime": "source_mtime",
        "days_lag": "source_days_lag",
        "status": "source_status",
    }
    cln_cols = {
        "path": "cleaned_path",
        "size_mb": "cleaned_size_mb",
        "mtime": "cleaned_mtime",
        "days_lag": "cleaned_days_lag",
        "status": "cleaned_status",
    }

    if not src.empty:
        src2 = src[["item", "subitem"] + list(src_cols.keys())].rename(columns=src_cols)
    else:
        src2 = pd.DataFrame(columns=["item", "subitem"] + list(src_cols.values()))

    if not cln.empty:
        cln2 = cln[["item", "subitem"] + list(cln_cols.keys())].rename(columns=cln_cols)
    else:
        cln2 = pd.DataFrame(columns=["item", "subitem"] + list(cln_cols.values()))

    merged = pd.merge(
        src2,
        cln2,
        on=["item", "subitem"],
        how="outer",
        validate="one_to_one",
    )

    def _relation(row) -> str:
        has_src = isinstance(row.get("source_path"), str) and row["source_path"] != ""
        has_cln = isinstance(row.get("cleaned_path"), str) and row["cleaned_path"] != ""
        if has_src and has_cln:
            return "both"
        elif has_src:
            return "source_only"
        elif has_cln:
            return "cleaned_only"
        else:
            return "none"

    merged["relation"] = merged.apply(_relation, axis=1)

    # mtime 差距：以天數表示（cleaned - source）
    if "source_mtime" in merged.columns and "cleaned_mtime" in merged.columns:
        merged["mtime_diff_days"] = (
            pd.to_datetime(merged["cleaned_mtime"]) - pd.to_datetime(merged["source_mtime"])
        ).dt.total_seconds() / 86400.0
    else:
        merged["mtime_diff_days"] = pd.NA

    return merged


def find_missing_cleaned(today: Optional[date] = None) -> pd.DataFrame:
    """
    找出「source 有但 cleaned 沒有」的 (item, subitem) 清單。
    """
    merged = merge_source_cleaned(today=today)
    if merged.empty:
        return merged
    mask = merged["relation"] == "source_only"
    return merged.loc[mask].copy()


def find_orphan_cleaned(today: Optional[date] = None) -> pd.DataFrame:
    """
    找出「cleaned 有但 source 沒有」的 (item, subitem)，通常代表：
        - 舊格式已不用的 cleaned 檔
        - 或是 source 路徑設定還沒補完
    """
    merged = merge_source_cleaned(today=today)
    if merged.empty:
        return merged
    mask = merged["relation"] == "cleaned_only"
    return merged.loc[mask].copy()


# ---------------------------------------------------------------------------
# 3. item 層級 summary
# ---------------------------------------------------------------------------

def summarize_cleaned_by_item(today: Optional[date] = None) -> pd.DataFrame:
    """
    針對 cleaned layer，彙總每個 item 的狀況：

        - n_subitems               : subitem 數量
        - n_files                  : 檔案數量（通常等於 n_subitems）
        - min_mtime / max_mtime
        - max_days_lag             : 最大落後天數
        - n_ok / n_lagging / n_stale
    """
    cln = scan_cleaned_files(today=today)
    if cln.empty:
        return cln

    grp = cln.groupby("item", as_index=False)

    summary = grp.agg(
        n_subitems=("subitem", "nunique"),
        n_files=("path", "count"),
        min_mtime=("mtime", "min"),
        max_mtime=("mtime", "max"),
        max_days_lag=("days_lag", "max"),
    )
    summary["min_mtime"] = pd.to_datetime(summary["min_mtime"]).dt.date
    summary["max_mtime"] = pd.to_datetime(summary["max_mtime"]).dt.date
    # 各狀態的筆數
    status_counts = (
        cln.groupby(["item", "status"])["path"]
        .count()
        .unstack(fill_value=0)
        .rename(
            columns={
                "OK": "n_ok",
                "LAGGING": "n_lagging",
                "STALE": "n_stale",
            }
        )
    )

    # 確保三個欄位都存在
    for col in ["n_ok", "n_lagging", "n_stale"]:
        if col not in status_counts.columns:
            status_counts[col] = 0

    summary = summary.join(status_counts, on="item")
    summary = summary.sort_values("item").reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# 4. 清理 log / error log 彙總（若有）
# ---------------------------------------------------------------------------

def load_cleaning_log() -> pd.DataFrame:
    """
    透過 LogMaintainer 載入 log.pkl 的原始內容。
    若 log.pkl 不存在，回傳空 DataFrame。
    """
    try:
        lm = LogMaintainer()
        return lm.df().copy()
    except FileNotFoundError:
        return pd.DataFrame()


def summarize_cleaning_log() -> pd.DataFrame:
    """
    盡量以彈性的方式，針對 log.pkl 做簡單彙總：

    - 若欄位中同時存在 ['item', 'subitem', 'status']：
        → 以這三個欄位 groupby 計數。

    - 否則：回傳原始 log，交由使用者自行觀察。
    """
    log = load_cleaning_log()
    if log.empty:
        return log

    required_cols = {"item", "subitem", "status"}
    if required_cols.issubset(log.columns):
        grouped = (
            log.groupby(list(required_cols))
            .size()
            .rename("count")
            .reset_index()
            .sort_values(
                ["item", "subitem", "status", "count"],
                ascending=[True, True, True, False],
            )
        )
        return grouped

    # 欄位名稱不符合預期，就直接回傳原始 DataFrame

    return log


def load_error_log() -> pd.DataFrame:
    """
    載入 errorlog.pkl 的原始內容；若不存在則回傳空 DataFrame。
    不在此階段做欄位解析，只提供原樣資料供檢視。
    """
    path = Path(dbpath_errorlog)
    if not path.exists():
        return pd.DataFrame()
    try:
        obj = pickleio(path=path, mode="load")
    except Exception:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 5. 對外快速介面（給 Notebook / 其他模組用）
# ---------------------------------------------------------------------------

def quick_inventory_summary(today: Optional[date] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    回傳三個 DataFrame：

        cleaned_detail  : cleaned 檔案層級明細
        item_summary    : cleaned 依 item 匯總
        relation_status : source vs cleaned 對比結果

    方便在 Notebook / 互動環境中直接查看。

    用法：
        cleaned_detail, item_summary, relation = quick_inventory_summary()
    """
    cleaned_detail = scan_cleaned_files(today=today)
    item_summary = summarize_cleaned_by_item(today=today)
    relation_status = merge_source_cleaned(today=today)
    return cleaned_detail, item_summary, relation_status


def get_twse_status(
    today: Optional[date] = None,
    include_log: bool = True,
    include_errorlog: bool = True,
) -> Dict[str, Any]:
    """
    提供一個整合好的「倉庫現況」字典，方便其他前端（例如 user_lab 的 script）使用。
    """
    if today is None:
        today = date.today()

    cleaned_detail, item_summary, relation_status = quick_inventory_summary(today=today)
    missing = find_missing_cleaned(today=today)
    orphan = find_orphan_cleaned(today=today)

    log_summary = summarize_cleaning_log() if include_log else pd.DataFrame()
    error_log = load_error_log() if include_errorlog else pd.DataFrame()

    return {
        "today": today,
        "cleaned_detail": cleaned_detail,
        "item_summary": item_summary,
        "relation_status": relation_status,
        "missing": missing,
        "orphan": orphan,
        "log_summary": log_summary,
        "error_log": error_log,
    }


# ---------------------------------------------------------------------------
# 6. 簡易 CLI 報表（取代舊的 warehouse_report）
# ---------------------------------------------------------------------------

def _format_title(text: str) -> str:
    line = "=" * len(text)
    return f"{line}\n{text}\n{line}"


def run_report(
    today: Optional[date] = None,
    show_log: bool = True,
    show_detail: bool = False,
    max_rows: int = 30,
) -> None:
    """
    在終端機輸出 TWSE 倉庫總覽報表。

    參數
    ----
    today : date, optional
        報表日期，預設為今天。
    show_log : bool
        是否顯示清理 log / error log 摘要。
    show_detail : bool
        是否加印 cleaned / relation 狀態的明細（只印前 max_rows 列）。
    max_rows : int
        明細最多顯示的列數。
    """
    status = get_twse_status(today=today, include_log=show_log, include_errorlog=show_log)
    today = status["today"]
    cleaned_detail = status["cleaned_detail"]
    item_summary = status["item_summary"]
    relation_status = status["relation_status"]
    missing = status["missing"]
    orphan = status["orphan"]
    log_summary = status["log_summary"]
    error_log = status["error_log"]

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)

    print(_format_title(f"TWSE 倉庫總覽（截至 {today}）"))

    # 1. item 層級 summary
    print("\n[1] Cleaned 依 item 彙總：")
    if item_summary.empty:
        print("  （尚無 cleaned 資料）")
    else:
        print(item_summary.to_string(index=False))

    # 2. source vs cleaned 缺漏
    print("\n[2] Source vs Cleaned 缺漏檢查：")
    if missing.empty and orphan.empty:
        print("  ✔ source / cleaned 結構一致，沒有缺漏或孤兒檔。")
    else:
        if not missing.empty:
            print("\n  2.1 source 有但 cleaned 沒有（缺少清理或路徑設定）：")
            cols = [c for c in ["item", "subitem", "source_mtime", "source_status"] if c in missing.columns]
            print(missing[cols].to_string(index=False))
        if not orphan.empty:
            print("\n  2.2 cleaned 有但 source 沒有（可能為舊格式或路徑未設定）：")
            cols = [c for c in ["item", "subitem", "cleaned_mtime", "cleaned_status"] if c in orphan.columns]
            print(orphan[cols].to_string(index=False))

    # 2.x 明細（選配）
    if show_detail:
        print(f"\n[2.3] Source vs Cleaned 對比明細（前 {max_rows} 列）：")
        if relation_status.empty:
            print("  （沒有任何 source / cleaned 資料）")
        else:
            print(relation_status.head(max_rows).to_string(index=False))

        print(f"\n[1.2] Cleaned 檔案層級明細（前 {max_rows} 列）：")
        if cleaned_detail.empty:
            print("  （沒有 cleaned 檔案）")
        else:
            print(cleaned_detail.head(max_rows).to_string(index=False))

    # 3. 清理 log / error log 摘要
    if show_log:
        print("\n[3] 清理 log 摘要（若 log.pkl 存在）：")
        if log_summary.empty:
            print("  （找不到 log.pkl 或內容為空）")
        else:
            head = log_summary.head(max_rows)
            print(head.to_string(index=False))
            if len(log_summary) > max_rows:
                print(f"\n  ... 共 {len(log_summary)} 筆，只顯示前 {max_rows} 筆。")

        print("\n[4] error log 摘要（若 errorlog.pkl 存在）：")
        if error_log.empty:
            print("  （找不到 errorlog.pkl 或內容為空）")
        else:
            # 只列出前幾列與欄位名稱，避免內容過長
            print("  欄位：", ", ".join(map(str, error_log.columns)))
            print("\n  內容（前幾列）：")
            print(error_log.head(max_rows).to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TWSE 倉庫管理員報表")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="不要顯示清理 log / error log 摘要",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="顯示 cleaned / relation 的明細（前 N 列）",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=30,
        help="明細列數上限（預設 30）",
    )
    args = parser.parse_args()

    run_report(
        today=None,
        show_log=not args.no_log,
        show_detail=args.detail,
        max_rows=args.max_rows,
    )
