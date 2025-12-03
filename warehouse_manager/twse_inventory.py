# -*- coding: utf-8 -*-
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
import datetime

import pandas as pd

from config.paths import (
    dbpath_cleaned,
    dbpath_source,
    dbpath_errorlog,
)

from config.conf import collection as TWSE_COLLECTION

from schema_utils import (
    scan_pkl_tree,
    add_days_lag,
    add_status_by_lag,
    LogMaintainer,
)
from StevenTricks.io.file_utils import pickleio


# ---------------------------------------------------------------------------
# 1. 檔案層級掃描
# ---------------------------------------------------------------------------

def _is_schema_path(path: str) -> bool:
    """
    根據檔名 / 路徑判斷是否為 'schema 類' 檔案，而非實際資料檔。

    規則（保守）：
        - 主檔名含有 'schema'
        - 主檔名以 '_schema' 結尾
        - 路徑中有資料夾名稱叫 'schema'
    """
    if not isinstance(path, str) or not path:
        return False

    p = Path(path)
    stem = p.stem.lower()

    if "schema" in stem or stem.endswith("_schema"):
        return True

    if any(part.lower() == "schema" for part in p.parts):
        return True

    return False


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
        - status      (OK / LAGGING / STALE)
        - is_schema   (是否為 schema 類檔案)
        - file_role   ("data" / "schema")
    """
    df = scan_pkl_tree(Path(dbpath_cleaned), layer="cleaned")
    if df.empty:
        return df

    # 標記 schema 類 cleaned（之後會從 pair 計算裡排除，不當成孤兒）
    df["is_schema"] = df["path"].map(_is_schema_path)
    df["file_role"] = df["is_schema"].map(lambda x: "schema" if x else "data")

    df = add_days_lag(df, date_col="mtime", today=today)
    df = add_status_by_lag(df, lag_col="days_lag")
    return df


def scan_source_files(today: Optional[date] = None) -> pd.DataFrame:
    """
    掃描 source DB 資料夾（爬蟲原始檔），欄位同 scan_cleaned_files。

    目前假設 source 沒有 schema 檔，全部當成 data。
    """
    df = scan_pkl_tree(Path(dbpath_source), layer="source")
    if df.empty:
        return df

    df["is_schema"] = False
    df["file_role"] = "data"

    df = add_days_lag(df, date_col="mtime", today=today)
    df = add_status_by_lag(df, lag_col="days_lag")
    return df


def summarize_source_by_item(today: Optional[date] = None) -> pd.DataFrame:
    """
    針對 source layer，彙總每個 item 的狀況：

        - n_subitems               : subitem 數量
        - n_files                  : 檔案數量
        - min_mtime / max_mtime    : 最舊 / 最新修改日期
        - max_days_lag             : 最大落後天數
        - n_ok / n_lagging / n_stale: 各狀態筆數
    """
    df = scan_source_files(today=today)
    if df.empty:
        return df

    grp = df.groupby("item", as_index=False).agg(
        n_subitems=("subitem", "nunique"),
        n_files=("path", "count"),
        min_mtime=("mtime", "min"),
        max_mtime=("mtime", "max"),
        max_days_lag=("days_lag", "max"),
    )
    grp["min_mtime"] = pd.to_datetime(grp["min_mtime"]).dt.date
    grp["max_mtime"] = pd.to_datetime(grp["max_mtime"]).dt.date

    status_counts = (
        df.groupby(["item", "status"])["path"]
        .count()
        .unstack(fill_value=0)
        .rename(columns={"OK": "n_ok", "LAGGING": "n_lagging", "STALE": "n_stale"})
    )

    for col in ["n_ok", "n_lagging", "n_stale"]:
        if col not in status_counts.columns:
            status_counts[col] = 0

    grp = grp.join(status_counts, on="item")
    grp = grp.sort_values("item").reset_index(drop=True)
    return grp


# ---------------------------------------------------------------------------
# 2. source vs cleaned 對比
# ---------------------------------------------------------------------------

def merge_source_cleaned(today: Optional[date] = None) -> pd.DataFrame:
    """
    以 (item, subitem) 為主鍵，把 source / cleaned 的檔案層級資訊做 outer join。

    只針對「資料檔」做對應，schema 檔不參與 pair 計算，
    避免被誤判成 cleaned_only 孤兒。
    """
    src = scan_source_files(today=today)
    cln = scan_cleaned_files(today=today)

    # 僅使用「資料檔」來做 source/cleaned 對應
    if "file_role" in src.columns:
        src = src[src["file_role"] != "schema"]
    if "file_role" in cln.columns:
        cln = cln[cln["file_role"] != "schema"]

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
    """找出「source 有但 cleaned 沒有」的 (item, subitem) 清單。"""
    merged = merge_source_cleaned(today=today)
    if merged.empty:
        return merged
    return merged.loc[merged["relation"] == "source_only"].copy()


def find_orphan_cleaned(today: Optional[date] = None) -> pd.DataFrame:
    """
    找出「cleaned 有但 source 沒有」的 (item, subitem)，通常代表：
        - 舊格式已不用的 cleaned 檔
        - 或是 source 路徑設定還沒補完
    schema 類 cleaned 不會出現在這裡（已在 merge_source_cleaned 時排除）。
    """
    merged = merge_source_cleaned(today=today)
    if merged.empty:
        return merged
    return merged.loc[merged["relation"] == "cleaned_only"].copy()


def prune_orphan_cleaned(today: Optional[date] = None, dry_run: bool = True) -> pd.DataFrame:
    """
    刪除「只有 cleaned、沒有對應 source」的檔案（真孤兒）。

    today : Optional[date]
        與 merge_source_cleaned / find_orphan_cleaned 一致，用來限制掃描範圍。
    dry_run : bool
        True  : 只回傳要刪除的清單，不實際刪檔（預設）
        False : 真的把 cleaned_path 指向的檔案刪掉。
    """
    orphan = find_orphan_cleaned(today=today)
    if orphan.empty:
        return orphan

    if dry_run:
        return orphan

    for _, row in orphan.iterrows():
        path = row.get("cleaned_path")
        if isinstance(path, str) and path:
            p = Path(path)
            if p.exists():
                try:
                    p.unlink()
                except Exception as e:
                    print(f"[prune_orphan_cleaned] 刪除失敗：{p} → {e}")

    return orphan


# ---------------------------------------------------------------------------
# 3. item 層級 summary
# ---------------------------------------------------------------------------

def summarize_cleaned_by_item(today: Optional[date] = None) -> pd.DataFrame:
    """
    針對 cleaned layer，彙總每個 item 的狀況（只看資料檔）：

        - n_subitems               : 資料檔的 subitem 數量
        - n_files                  : 資料檔檔案數量
        - min_mtime / max_mtime    : 資料檔的最舊 / 最新 mtime
        - max_days_lag             : 資料檔的最大落後天數
        - n_ok / n_lagging / n_stale: 資料檔各狀態筆數
        - n_schema_files           : schema 類 cleaned 檔數量
        - has_schema               : 是否至少有一個 schema 檔
    """
    cln = scan_cleaned_files(today=today)
    if cln.empty:
        return cln

    if "is_schema" in cln.columns:
        data = cln[~cln["is_schema"]].copy()
        schema = cln[cln["is_schema"]].copy()
    else:
        data = cln.copy()
        schema = cln.iloc[0:0].copy()

    # --- 資料檔 summary ---
    if data.empty:
        items = schema["item"].drop_duplicates()
        summary = pd.DataFrame({"item": items})
        summary["n_subitems"] = 0
        summary["n_files"] = 0
        summary["min_mtime"] = pd.NaT
        summary["max_mtime"] = pd.NaT
        summary["max_days_lag"] = pd.NA
        status_counts = pd.DataFrame(
            0, index=items, columns=["n_ok", "n_lagging", "n_stale"]
        )
    else:
        grp = data.groupby("item", as_index=False)
        summary = grp.agg(
            n_subitems=("subitem", "nunique"),
            n_files=("path", "count"),
            min_mtime=("mtime", "min"),
            max_mtime=("mtime", "max"),
            max_days_lag=("days_lag", "max"),
        )
        summary["min_mtime"] = pd.to_datetime(summary["min_mtime"]).dt.date
        summary["max_mtime"] = pd.to_datetime(summary["max_mtime"]).dt.date

        status_counts = (
            data.groupby(["item", "status"])["path"]
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

    for col in ["n_ok", "n_lagging", "n_stale"]:
        if col not in status_counts.columns:
            status_counts[col] = 0

    summary = summary.join(status_counts, on="item")

    # --- schema 檔數量 & flag ---
    if not schema.empty:
        schema_cnt = (
            schema.groupby("item")["path"]
            .count()
            .rename("n_schema_files")
        )
    else:
        schema_cnt = pd.Series(dtype="int64", name="n_schema_files")

    summary = summary.join(schema_cnt, on="item")
    summary["n_schema_files"] = summary["n_schema_files"].fillna(0).astype(int)
    summary["has_schema"] = summary["n_schema_files"] > 0

    summary = summary.sort_values("item").reset_index(drop=True)
    return summary


def get_expected_items_df() -> pd.DataFrame:
    """
    從 config.conf.collection 產生「設計上應存在的 TWSE item」列表。

    欄位：
        - item
        - freq
        - datemin
        - n_expected_subitems
        - code
    """
    rows: list[dict] = []
    for item_name, cfg in TWSE_COLLECTION.items():
        freq = cfg.get("freq")
        datemin = cfg.get("datemin")
        subtitles = cfg.get("subtitle") or []
        code = cfg.get("code")

        try:
            datemin_date = (
                pd.to_datetime(datemin, errors="coerce").date()
                if datemin
                else pd.NaT
            )
        except Exception:
            datemin_date = pd.NaT

        rows.append(
            dict(
                item=item_name,
                freq=freq,
                datemin=datemin_date,
                n_expected_subitems=len(subtitles),
                code=code,
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=["item", "freq", "datemin", "n_expected_subitems", "code"]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("item").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 4. 清理 log / error log 彙總（若有）
# ---------------------------------------------------------------------------

def load_cleaning_log() -> pd.DataFrame:
    """透過 LogMaintainer 載入 log.pkl 的原始內容。若不存在則回傳空 DataFrame。"""
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


def summarize_error_log(raw: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    將 errorlog.pkl 的「寬表」轉成一列一錯誤的「長表」，再做簡單統計。

    回傳：
        error_flat   : 每列一筆錯誤
        reason_stats : 依 (error_reason, item) 統計筆數
    """
    if raw is None:
        raw = load_error_log()

    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    records: list[Dict[str, Any]] = []

    for dt, row in raw.iterrows():
        for item, cell in row.items():
            if cell is None:
                continue
            try:
                import math
                if isinstance(cell, float) and math.isnan(cell):
                    continue
            except Exception:
                pass

            if isinstance(cell, list):
                errors = cell
            else:
                errors = [cell]

            for e in errors:
                if not isinstance(e, dict):
                    e = {"raw": e}

                reason = (
                    e.get("error_reason")
                    or e.get("err_type")
                    or e.get("error_type")
                    or e.get("errormessage3")
                    or e.get("errormessage1")
                    or "unknown"
                )
                msg = e.get("error_message") or e.get("errormessage2") or ""
                log_time = e.get("log_time")

                if isinstance(log_time, (datetime.datetime, datetime.date)):
                    log_date = log_time.date()
                else:
                    log_date = None

                rec = {
                    "date": dt,
                    "item": item,
                    "error_reason": reason,
                    "error_message": msg,
                    "requeststatus": e.get("requeststatus"),
                    "exception_type": e.get("exception_type"),
                    "log_time": log_time,
                    "log_date": log_date,
                }
                records.append(rec)

    error_flat = pd.DataFrame.from_records(records)
    if error_flat.empty:
        return error_flat, pd.DataFrame()

    error_flat["date"] = pd.to_datetime(error_flat["date"], errors="coerce").dt.date
    if "log_date" in error_flat.columns:
        error_flat["log_date"] = pd.to_datetime(error_flat["log_date"], errors="coerce").dt.date

    reason_stats = (
        error_flat.groupby(["error_reason", "item"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return error_flat, reason_stats


# ---------------------------------------------------------------------------
# 5. 對外快速介面（給 Notebook / 其他模組用）
# ---------------------------------------------------------------------------

def quick_inventory_summary(today: Optional[date] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    回傳三個 DataFrame：

        cleaned_detail  : cleaned 檔案層級明細
        item_summary    : cleaned 依 item 匯總
        relation_status : source vs cleaned 對比結果

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


def build_twse_dashboard(
    today: Optional[date] = None,
    include_log: bool = True,
    include_errorlog: bool = True,
    item_top_n: int = 20,
    top_rows: int = 50,
) -> Dict[str, Any]:
    """
    建立完整的 TWSE 倉庫 dashboard，集中所有「會用到的 DataFrame」。
    """
    status = get_twse_status(
        today=today,
        include_log=include_log,
        include_errorlog=include_errorlog,
    )
    today = status["today"]

    cleaned_detail = status["cleaned_detail"]
    item_summary = status["item_summary"].copy()
    relation_status = status["relation_status"]
    missing = status["missing"].copy()
    orphan = status["orphan"].copy()
    log_summary = status["log_summary"]
    error_log = status["error_log"]

    # schema 檔資訊：從 cleaned_detail 中拆出來
    if cleaned_detail is not None and not cleaned_detail.empty and "file_role" in cleaned_detail.columns:
        schema_detail = cleaned_detail[cleaned_detail["file_role"] == "schema"].copy()
        schema_item_summary = (
            schema_detail.groupby("item", as_index=False)
            .agg(n_schema_files=("path", "count"))
        )
    else:
        schema_detail = pd.DataFrame()
        schema_item_summary = pd.DataFrame(columns=["item", "n_schema_files"])

    # 補上 source / source item summary
    source_detail = scan_source_files(today=today)
    source_item_summary = summarize_source_by_item(today=today)
    cleaned_item_summary = item_summary

    # 小工具：把時間欄位轉成 date（只到日）
    def _to_date(df: pd.DataFrame, cols: list[str]) -> None:
        if df is None or df.empty:
            return
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce").dt.date

    _to_date(missing, ["source_mtime", "cleaned_mtime"])
    _to_date(orphan, ["source_mtime", "cleaned_mtime"])

    # 整體統計 overall_summary（只看 data 檔的 pair）
    rel = relation_status
    if rel is None or rel.empty:
        n_pairs_total = n_pairs_source = n_pairs_cleaned = 0
        n_both = n_source_only = n_cleaned_only = 0
    else:
        has_src = rel["source_path"].notna() & (rel["source_path"] != "")
        has_cln = rel["cleaned_path"].notna() & (rel["cleaned_path"] != "")
        n_pairs_total = int(len(rel))
        n_pairs_source = int(has_src.sum())
        n_pairs_cleaned = int(has_cln.sum())
        n_both = int((rel["relation"] == "both").sum())
        n_source_only = int((rel["relation"] == "source_only").sum())
        n_cleaned_only = int((rel["relation"] == "cleaned_only").sum())

    n_items_cleaned = int(item_summary["item"].nunique()) if not item_summary.empty else 0

    overall_summary = pd.DataFrame(
        [
            dict(
                as_of=today,
                n_items_cleaned=n_items_cleaned,
                n_source_files=int(source_detail.shape[0]),
                n_cleaned_files=int(cleaned_detail.shape[0]),
                n_pairs_total=n_pairs_total,
                n_pairs_source=n_pairs_source,
                n_pairs_cleaned=n_pairs_cleaned,
                n_pairs_both=n_both,
                n_pairs_source_only=n_source_only,
                n_pairs_cleaned_only=n_cleaned_only,
                n_missing_pairs=int(len(missing)),
                n_orphan_pairs=int(len(orphan)),
            )
        ]
    )

    # 依 config.collection 檢查「應有 item」的覆蓋狀況
    try:
        expected_item_master = get_expected_items_df()
    except Exception:
        expected_item_master = pd.DataFrame(
            columns=["item", "freq", "datemin", "n_expected_subitems", "code"]
        )

    if expected_item_master.empty:
        expected_item_status = pd.DataFrame()
        unexpected_items = pd.DataFrame(columns=["item", "in_source", "in_cleaned"])
    else:
        if source_item_summary is None or source_item_summary.empty:
            src = pd.DataFrame(columns=["item"])
        else:
            src = source_item_summary.rename(
                columns={
                    "n_subitems": "source_n_subitems",
                    "n_files": "source_n_files",
                    "min_mtime": "source_min_mtime",
                    "max_mtime": "source_max_mtime",
                    "max_days_lag": "source_max_days_lag",
                    "n_ok": "source_n_ok",
                    "n_lagging": "source_n_lagging",
                    "n_stale": "source_n_stale",
                }
            )

        if cleaned_item_summary is None or cleaned_item_summary.empty:
            cln = pd.DataFrame(columns=["item"])
        else:
            cln = cleaned_item_summary.rename(
                columns={
                    "n_subitems": "cleaned_n_subitems",
                    "n_files": "cleaned_n_files",
                    "min_mtime": "cleaned_min_mtime",
                    "max_mtime": "cleaned_max_mtime",
                    "max_days_lag": "cleaned_max_days_lag",
                    "n_ok": "cleaned_n_ok",
                    "n_lagging": "cleaned_n_lagging",
                    "n_stale": "cleaned_n_stale",
                    "n_schema_files": "cleaned_n_schema_files",
                    "has_schema": "has_schema",
                }
            )

        expected_item_status = (
            expected_item_master
            .merge(src, on="item", how="left")
            .merge(cln, on="item", how="left")
        )

        if "source_n_files" not in expected_item_status.columns:
            expected_item_status["source_n_files"] = 0
        if "cleaned_n_files" not in expected_item_status.columns:
            expected_item_status["cleaned_n_files"] = 0

        expected_item_status["has_source"] = (
            expected_item_status["source_n_files"].fillna(0).astype(int) > 0
        )
        expected_item_status["has_cleaned"] = (
            expected_item_status["cleaned_n_files"].fillna(0).astype(int) > 0
        )
        expected_item_status["missing_source"] = ~expected_item_status["has_source"]
        expected_item_status["missing_cleaned"] = ~expected_item_status["has_cleaned"]

        # schema 存在與否（這裡順便修掉 pandas 的 FutureWarning）
        if "has_schema" not in expected_item_status.columns:
            expected_item_status["has_schema"] = False

        s = expected_item_status["has_schema"]
        s = s.infer_objects(copy=False)   # 先把 object 裡的 bool 推回 bool
        s = s.fillna(False).astype(bool)
        expected_item_status["has_schema"] = s
        expected_item_status["missing_schema"] = ~s

        # 找出「實際出現但不在 config.collection」的 item
        expected_set = set(expected_item_master["item"].unique())
        src_set = (
            set(source_item_summary["item"].unique())
            if source_item_summary is not None and not source_item_summary.empty
            else set()
        )
        cln_set = (
            set(cleaned_item_summary["item"].unique())
            if cleaned_item_summary is not None and not cleaned_item_summary.empty
            else set()
        )
        unexpected = sorted((src_set | cln_set) - expected_set)

        rows = []
        for itm in unexpected:
            rows.append(
                dict(
                    item=itm,
                    in_source=itm in src_set,
                    in_cleaned=itm in cln_set,
                )
            )
        unexpected_items = pd.DataFrame(rows)

        # 把 expected vs actual 的統計加進 overall_summary
        n_expected = int(expected_item_master.shape[0])
        n_expected_src = int(expected_item_status["has_source"].sum())
        n_expected_cln = int(expected_item_status["has_cleaned"].sum())
        n_missing_src = int(expected_item_status["missing_source"].sum())
        n_missing_cln = int(expected_item_status["missing_cleaned"].sum())
        n_schema_ok = int(expected_item_status["has_schema"].sum())
        n_schema_missing = int(expected_item_status["missing_schema"].sum())

        n_unexp_src = int(unexpected_items["in_source"].sum()) if not unexpected_items.empty else 0
        n_unexp_cln = int(unexpected_items["in_cleaned"].sum()) if not unexpected_items.empty else 0

        if overall_summary is None or overall_summary.empty:
            overall_summary = pd.DataFrame([dict(as_of=today)])

        for col, val in [
            ("n_items_expected", n_expected),
            ("n_items_expected_in_source", n_expected_src),
            ("n_items_expected_in_cleaned", n_expected_cln),
            ("n_items_expected_missing_source", n_missing_src),
            ("n_items_expected_missing_cleaned", n_missing_cln),
            ("n_items_expected_with_schema", n_schema_ok),
            ("n_items_expected_missing_schema", n_schema_missing),
            ("n_items_unexpected_source", n_unexp_src),
            ("n_items_unexpected_cleaned", n_unexp_cln),
        ]:
            overall_summary[col] = val

    # 每個 item 的 relation 統計
    if relation_status is not None and not relation_status.empty:
        relation_counts = (
            relation_status.groupby(["item", "relation"])
            .size()
            .rename("count")
            .reset_index()
            .pivot(index="item", columns="relation", values="count")
            .fillna(0)
            .astype(int)
            .rename_axis(None, axis=1)
            .reset_index()
        )
    else:
        relation_counts = pd.DataFrame()

    # lagging 排名
    if not item_summary.empty:
        lag_item = item_summary.sort_values(
            "max_days_lag", ascending=False, na_position="last"
        ).reset_index(drop=True)
        lag_item_top = lag_item.head(item_top_n).copy()
    else:
        lag_item = pd.DataFrame()
        lag_item_top = pd.DataFrame()

    missing_top = missing.head(top_rows).copy()
    orphan_top = orphan.head(top_rows).copy()

    # error_log 攤平與統計
    if include_errorlog and error_log is not None and not error_log.empty:
        error_flat, error_reason_summary = summarize_error_log(raw=error_log)
        if not error_flat.empty:
            if "date" in error_flat.columns:
                error_date_summary = (
                    error_flat.groupby("date", dropna=False)
                    .size()
                    .reset_index(name="n_errors")
                    .sort_values("date")
                )
            else:
                error_date_summary = pd.DataFrame()

            if "item" in error_flat.columns:
                error_item_summary = (
                    error_flat.groupby("item", dropna=False)
                    .size()
                    .reset_index(name="n_errors")
                    .sort_values("n_errors", ascending=False)
                )
            else:
                error_item_summary = pd.DataFrame()
        else:
            error_date_summary = pd.DataFrame()
            error_item_summary = pd.DataFrame()
    else:
        error_flat = pd.DataFrame()
        error_reason_summary = pd.DataFrame()
        error_date_summary = pd.DataFrame()
        error_item_summary = pd.DataFrame()

    return dict(
        today=today,
        source_detail=source_detail,
        cleaned_detail=cleaned_detail,
        item_summary=item_summary,
        relation_status=relation_status,
        missing=missing,
        orphan=orphan,
        log_summary=log_summary,
        error_log=error_log,
        overall_summary=overall_summary,
        source_item_summary=source_item_summary,
        cleaned_item_summary=cleaned_item_summary,
        relation_counts=relation_counts,
        lag_item=lag_item,
        lag_item_top=lag_item_top,
        missing_top=missing_top,
        orphan_top=orphan_top,
        error_log_flat=error_flat,
        error_reason_summary=error_reason_summary,
        error_date_summary=error_date_summary,
        error_item_summary=error_item_summary,
        schema_detail=schema_detail,
        schema_item_summary=schema_item_summary,
        expected_item_master=expected_item_master,
        expected_item_status=expected_item_status,
        unexpected_items=unexpected_items,
    )


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

    用法（專案根目錄）：
        python -m warehouse_manager.twse_inventory
    或從其他 .py：
        from warehouse_manager.twse_inventory import run_report
        run_report()
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

    print("\n[1] Cleaned 依 item 彙總：")
    if item_summary.empty:
        print("  （尚無 cleaned 資料）")
    else:
        print(item_summary.to_string(index=False))

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


if __name__ == "__main__":
    run_report()
