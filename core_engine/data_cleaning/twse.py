#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TWSE 清理器：合併自 data_cleaning/cleaner_main.py + cleaning_utils.py

特性
- 可 CLI 與 import 雙用
- 預設 DEBUG logging（不需要 --debug）
- 嚴格錯誤策略：清理過程遇錯「立刻中止」，不跳過、不寫 .csv/.txt；把出錯的資料值/型別、欄位、子表、檔案等詳列，方便你立即補清理規則
- 每個函式可單獨測試
- 重複/冗餘邏輯已精簡（欄名歸一處理等）
"""
import argparse
import logging
from os import makedirs
from os.path import exists, join, basename
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime
from dataclasses import dataclass, field
import time
import math
import sys
import pandas as pd
import shutil
import re
import unicodedata

# ---- StevenTricks 與 config ----
from config.conf import fields_span, dropcol, key_set
from config.col_rename import colname_dic, transtonew_col
from config.col_format import numericol, datecol
from StevenTricks.io.file_utils import pickleio, PathWalk_df
from StevenTricks.core.convert_utils import safe_replace, safe_numeric_convert, stringtodate
from StevenTricks.db.internal_db import DBPkl
from config.paths import (
    dbpath_source as CLOUD_DBPATH_SOURCE,
    dbpath_cleaned as CLOUD_DBPATH_CLEANED,
    dbpath_cleaned_log as CLOUD_DBPATH_CLEANED_LOG,
    db_local_root,
)
from config.conf import collection
from StevenTricks.io.staging import staging_path


dbpath_source = CLOUD_DBPATH_SOURCE
dbpath_cleaned = CLOUD_DBPATH_CLEANED
dbpath_cleaned_log = CLOUD_DBPATH_CLEANED_LOG

LOCAL_DB_ROOT = db_local_root
LOCAL_DBPATH_SOURCE = LOCAL_DB_ROOT / "source"
LOCAL_DBPATH_CLEANED = LOCAL_DB_ROOT / "cleaned"
LOCAL_DBPATH_CLEANED_LOG = LOCAL_DBPATH_CLEANED / "log.pkl"


DEBUG_LAST_DF: Optional[pd.DataFrame] = None
DEBUG_LAST_CONTEXT: Dict[str, Any] = {}

UPDATE_OLD_NON_DAILY: bool = False

_root = logging.getLogger()
if not _root.handlers:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JOB_STATE_COLUMNS = [
    "file",
    "path",
    "dir",
    "hash",
    "source_mtime",
    "source_size",
    "status",
    "date",
    "item",
    "last_processed_at",
]


def _overwrite_line(msg: str) -> None:
    sys.stdout.write("\r" + msg[:240])
    sys.stdout.flush()

def _overwrite_line_end() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


def _estimate_remaining_by_job_state(files_df: pd.DataFrame, job_state: pd.DataFrame) -> int:
    if files_df is None or files_df.empty:
        return 0

    if job_state is None or job_state.empty or "path" not in job_state.columns:
        return int(len(files_df))

    js = job_state[["path", "status"]].copy()
    js["path"] = js["path"].astype(str)

    merged = files_df[["path"]].copy()
    merged["path"] = merged["path"].astype(str)

    merged = merged.merge(js, on="path", how="left")
    pending_mask = merged["status"].isna() | (merged["status"].astype(str) != "success")
    return int(pending_mask.sum())


def _calc_file_state(path: str) -> Dict[str, Any]:
    p = Path(path)
    st = p.stat()
    size = st.st_size
    mtime = st.st_mtime
    fp = f"{size}-{int(mtime)}"

    return {
        "hash": fp,
        "source_size": size,
        "source_mtime": pd.Timestamp.fromtimestamp(mtime),
    }


def _needs_processing_exact(file_path: str, rec: Optional[pd.Series]) -> bool:
    state_now = _calc_file_state(file_path)
    fp_now = state_now["hash"]
    mtime_now = state_now["source_mtime"]

    if rec is None:
        return True

    rec_status = rec.get("status")
    rec_hash = rec.get("hash")
    rec_mtime = rec.get("source_mtime")

    if (
        rec_status == "success"
        and pd.notna(rec_hash)
        and str(rec_hash) == str(fp_now)
        and pd.notna(rec_mtime)
        and pd.Timestamp(rec_mtime) == mtime_now
    ):
        return False

    return True


def _has_any_pending_exact(files_df: pd.DataFrame, job_state: pd.DataFrame) -> bool:
    if files_df is None or files_df.empty:
        return False

    js = job_state.copy() if isinstance(job_state, pd.DataFrame) else pd.DataFrame(columns=JOB_STATE_COLUMNS)
    if js.empty or "path" not in js.columns:
        js = pd.DataFrame(columns=JOB_STATE_COLUMNS)

    rec_map: Dict[str, pd.Series] = {}
    if not js.empty:
        for _, r in js.iterrows():
            rec_map[str(r.get("path"))] = r

    for _, row in files_df.iterrows():
        p = str(row["path"])
        rec = rec_map.get(p)
        if _needs_processing_exact(p, rec):
            return True
    return False


def _ensure_dir(p) -> None:
    makedirs(p, exist_ok=True)


def _clear_local_paths() -> None:
    targets = [LOCAL_DBPATH_SOURCE, LOCAL_DBPATH_CLEANED]
    for p in targets:
        try:
            if p.exists():
                shutil.rmtree(p)
                logger.info("已清空本機目錄：%s", p)
        except Exception as e:
            logger.warning("清空本機目錄失敗：%s (%s)", p, e)

    try:
        if LOCAL_DBPATH_CLEANED_LOG.exists():
            LOCAL_DBPATH_CLEANED_LOG.unlink()
            logger.info("已刪除本機 job_state log：%s", LOCAL_DBPATH_CLEANED_LOG)
    except Exception as e:
        logger.warning("刪除本機 job_state log 失敗：%s (%s)", p, e)


def _clear_staging_dirs(staging_root: Path) -> None:
    if not staging_root.exists():
        return

    for sub in staging_root.iterdir():
        if not sub.is_dir():
            continue
        if sub.name.startswith("staging_"):
            try:
                shutil.rmtree(sub)
                logger.info("已清除舊 staging 目錄：%s", sub)
            except Exception as e:
                logger.warning("清除 staging 目錄失敗：%s (%s)", sub, e)


def _extract_legacy_subtables(raw: dict) -> list[dict]:
    subtables: list[dict] = []

    for key in raw.keys():
        m = re.match(r"^fields(\d+)$", key)
        if not m:
            continue
        idx = int(m.group(1))

        fields = raw.get(f"fields{idx}")
        data = raw.get(f"data{idx}")
        title = raw.get(f"subtitle{idx}")

        if not fields or not data:
            continue

        subtables.append(
            {
                "idx": idx,
                "title": title,
                "fields": fields,
                "data": data,
            }
        )

    return subtables
def _get_span_cfg(item: str, subitem: str) -> Optional[dict]:
    """
    取得 span 設定（fields_span）：
    - 允許兩種寫法：
        1) fields_span[item][subitem]
        2) fields_span[subitem]
    - 支援兩種格式：
        A) 新版：{"groups": [{"size": 3, "prefix": "股票_"}, ...]}
        B) 舊版：{"股票": 3, "融資": 5, ...} 或 {"股票":{"size":3}, ...}
           也支援 size 的別名：n/span/len/count
    """
    by_item = (fields_span.get(item, {}) or {}).get(subitem)
    direct = fields_span.get(subitem)
    cfg = by_item or direct
    if not cfg:
        return None
    if not isinstance(cfg, dict):
        logger.warning(
            "[fields_span ignored] cfg not dict -> fallback frameup_safe | item=%s subitem=%s cfg_type=%s",
            item, subitem, type(cfg).__name__
        )
        return None

    # --- 新版格式 ---
    groups = cfg.get("groups")
    if isinstance(groups, list) and groups:
        for i, g in enumerate(groups):
            try:
                size = int(g.get("size", 0) or 0)
            except Exception:
                size = 0
            if not isinstance(g, dict) or size <= 0:
                logger.warning(
                    "[fields_span ignored] invalid group(size) -> fallback frameup_safe | item=%s subitem=%s group_idx=%d group=%r",
                    item, subitem, i, g
                )
                return None
        return cfg

    # --- 舊版格式 ---
    def _pick_size_from_dict(d: dict) -> int:
        for k in ("size", "n", "span", "len", "count"):
            if k in d:
                try:
                    return int(d.get(k) or 0)
                except Exception:
                    return 0
        # 若 dict 只有一個值且可轉 int，也吃
        if len(d) == 1:
            try:
                return int(list(d.values())[0] or 0)
            except Exception:
                return 0
        return 0

    legacy_keys = [k for k in cfg.keys() if k not in {"groups"}]
    if legacy_keys:
        legacy_groups = []
        ok = True
        for k in legacy_keys:
            v = cfg.get(k)

            prefix = f"{k}_"
            size = 0

            if isinstance(v, dict):
                size = _pick_size_from_dict(v)
                if "prefix" in v:
                    prefix = str(v.get("prefix") or prefix)
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                # e.g. ["股票_", 3] or ("股票_", 3)
                prefix = str(v[0] or prefix)
                try:
                    size = int(v[1] or 0)
                except Exception:
                    size = 0
            else:
                try:
                    size = int(v)
                except Exception:
                    size = 0

            if size <= 0:
                ok = False
                break

            legacy_groups.append({"size": size, "prefix": prefix})

        if ok and legacy_groups:
            logger.info(
                "[fields_span legacy] auto-convert -> groups | item=%s subitem=%s groups=%s",
                item, subitem, legacy_groups
            )
            return {"groups": legacy_groups}

    logger.warning(
        "[fields_span ignored] missing/invalid groups -> fallback frameup_safe | item=%s subitem=%s cfg_keys=%s",
        item, subitem, list(cfg.keys())
    )
    return None

def _is_partition_by_date_item(item: str) -> bool:
    cfg = collection.get(item) or {}
    freq = cfg.get("freq")
    if freq is None:
        return False
    s = str(freq).strip().upper()
    return s in {"D", "1D", "DAY", "DAILY"}


def _is_me_freq_item(item: str) -> bool:
    cfg = collection.get(item) or {}
    freq = cfg.get("freq")
    if freq is None:
        return False
    s = str(freq).strip().upper()
    return s == "ME"


def _is_non_daily_freq_item(item: str) -> bool:
    cfg = collection.get(item) or {}
    freq = cfg.get("freq")
    if freq is None:
        return False
    s = str(freq).strip().upper()
    return s not in {"D", "1D", "DAY", "DAILY"}


def _apply_non_daily_policy(
    df: pd.DataFrame,
    *,
    item: str,
    subitem: str,
    file_name: str,
    update_old_non_daily: bool,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not _is_non_daily_freq_item(item):
        return df
    if update_old_non_daily:
        return df
    if "date" not in df.columns:
        return df

    d = pd.to_datetime(df["date"]).dt.normalize()
    latest = d.max()
    before = len(df)
    out = df.loc[d == latest].copy()
    after = len(out)

    if after < before:
        logger.info(
            "非日頻只取最新 date（不回補舊資料）：item=%s subitem=%s latest=%s rows=%d->%d file=%s",
            item, subitem, str(latest.date()), before, after, file_name
        )
    return out


def _make_bucket_key(date_series: pd.Series, mode: str) -> pd.Series:
    mode = (mode or "all").lower()

    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series)

    if mode == "year":
        return date_series.dt.strftime("%Y")
    if mode == "quarter":
        return date_series.dt.to_period("Q").astype(str)
    if mode == "month":
        return date_series.dt.strftime("%Y-%m")
    if mode == "day":
        return date_series.dt.strftime("%Y-%m-%d")

    return pd.Series(["ALL"] * len(date_series), index=date_series.index)


class DataCleanError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        file: Optional[str] = None,
        item: Optional[str] = None,
        subitem: Optional[str] = None,
        column: Optional[Any] = None,
        value: Any = None,
        value_type: Optional[str] = None,
        date: Optional[str] = None,
        hint: Optional[str] = None,
    ):
        parts = [message]
        ctx = []
        if file: ctx.append(f"file={file}")
        if item: ctx.append(f"item={item}")
        if subitem: ctx.append(f"subitem={subitem}")
        if column is not None: ctx.append(f"column={repr(column)}")
        if value is not None: ctx.append(f"value={repr(value)}")
        if value_type: ctx.append(f"value_type={value_type}")
        if date: ctx.append(f"date={date}")
        if hint: ctx.append(f"hint={hint}")
        if ctx:
            parts.append(" | " + ", ".join(ctx))
        super().__init__("".join(parts))


def _match_subtitle_from_conf_by_contains(
    *,
    item: str,
    raw_title: Any,
    file_name: str,
) -> str:
    if raw_title is None:
        raise DataCleanError(
            "子表標題為 None，無法與 collection.subtitle 配對",
            file=file_name,
            item=item,
            value={"raw_title": raw_title},
        )

    title_norm = unicodedata.normalize("NFKC", str(raw_title)).strip()
    if not title_norm:
        raise DataCleanError(
            "子表標題為空字串，無法與 collection.subtitle 配對",
            file=file_name,
            item=item,
            value={"raw_title": raw_title},
        )

    cfg = collection.get(item) or {}
    subtitles_conf = cfg.get("subtitle")

    if not subtitles_conf:
        raise DataCleanError(
            "config.collection[item]['subtitle'] 未設定或為空，無法判定子表",
            file=file_name,
            item=item,
            value={"raw_title": raw_title, "title_norm": title_norm},
        )

    if isinstance(subtitles_conf, str):
        subtitles_conf = [subtitles_conf]

    hits: List[str] = []
    for s in subtitles_conf:
        s_norm = unicodedata.normalize("NFKC", str(s)).strip()
        if not s_norm:
            continue
        if s_norm in title_norm:
            hits.append(s)

    if len(hits) == 1:
        return hits[0]

    if len(hits) == 0:
        raise DataCleanError(
            "子表標題在 collection[item]['subtitle'] 中完全配對不到",
            file=file_name,
            item=item,
            value={"raw_title": raw_title, "title_norm": title_norm, "subtitle_list": list(subtitles_conf)},
        )

    raise DataCleanError(
        "子表標題在 collection[item]['subtitle'] 中配對到多個候選，無法唯一判定",
        file=file_name,
        item=item,
        value={"raw_title": raw_title, "title_norm": title_norm, "candidates": hits},
    )


def _normalize_cols(cols: List[str]) -> List[str]:
    mapped = [colname_dic.get(c, c) for c in cols]
    cleaned = [safe_replace(str(c), "</br>", "") for c in mapped]
    return cleaned
def _get_span_prefix_candidates(item: str, subitem: str) -> List[str]:
    cfg = (fields_span.get(item, {}) or {}).get(subitem) or fields_span.get(subitem) or {}
    if not isinstance(cfg, dict):
        return []
    # legacy keys (股票/融資/融券/其他)
    keys = [k for k in cfg.keys() if k not in {"groups"}]
    return [str(k).strip() for k in keys if str(k).strip()]


def _make_unique_by_suffix(cols: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        n = seen.get(c, 0) + 1
        seen[c] = n
        out.append(c if n == 1 else f"{c}__{n}")
    return out


def _repair_duplicate_columns(
    df: pd.DataFrame,
    *,
    file_name: str,
    item: str,
    subitem: str,
    table_name: str,
) -> pd.DataFrame:
    cols = list(map(str, df.columns))
    vc = pd.Series(cols).value_counts()
    dup = vc[vc > 1]
    if dup.empty:
        return df

    prefixes = _get_span_prefix_candidates(item, subitem)
    # 先試「用 prefixes 分塊加前綴」：最符合你這張表的結構
    if prefixes:
        g = len(prefixes)

        # 找第一個重複欄位出現的位置，假設此後是分組區
        first_dup_col = dup.index[0]
        try:
            start = cols.index(first_dup_col)
        except ValueError:
            start = 0

        tail = cols[start:]
        if g >= 2 and len(tail) % g == 0:
            chunk = len(tail) // g
            new_tail: List[str] = []
            for i in range(g):
                pref = prefixes[i]
                seg = tail[i * chunk : (i + 1) * chunk]
                new_tail.extend([f"{pref}_{c}" for c in seg])

            fixed = cols[:start] + new_tail
            if pd.Index(fixed).is_unique:
                logger.warning(
                    "欄名重複：已自動修復（加上分組前綴）| file=%s item=%s subitem=%s table=%s dup_cols=%s prefixes=%s",
                    file_name, item, subitem, table_name, dup.index.tolist(), prefixes
                )
                out = df.copy()
                out.columns = fixed
                return out

    # 再退一步：用 __2/__3 suffix 保證唯一（可寫入 DB，但你要接受欄意義較弱）
    fixed2 = _make_unique_by_suffix(cols)
    if pd.Index(fixed2).is_unique:
        logger.warning(
            "欄名重複：已以序號後綴修復（__2/__3...）| file=%s item=%s subitem=%s table=%s dup_cols=%s",
            file_name, item, subitem, table_name, dup.index.tolist()
        )
        out = df.copy()
        out.columns = fixed2
        return out

    # 理論上走不到這
    raise DataCleanError(
        "欄名重複且無法自動修復（不允許）",
        file=file_name, item=item, subitem=subitem,
        value={"table_name": table_name, "dup_cols": dup.index.tolist(), "dup_counts": dup.to_dict(), "n_cols": len(cols)},
        hint="請補 fields_span/groups 或調整 subtitle/欄位解析",
    )


def _list_source_pickles(root) -> pd.DataFrame:
    df = PathWalk_df(root, [], ["log"], [".DS_Store", "productlist"], [".pkl"])
    need_cols = {"file", "path", "dir"}
    have = set(df.columns)
    if not need_cols.issubset(have):
        if "path" not in df.columns:
            raise DataCleanError("PathWalk_df 缺少 path 欄")
        df["file"] = df["path"].map(lambda p: basename(p))
        df["dir"] = df["path"].map(lambda p: Path(p).parent.name)
    return df[["file", "path", "dir"]].copy()


def _load_job_state() -> pd.DataFrame:
    if not exists(dbpath_cleaned_log):
        return pd.DataFrame(columns=JOB_STATE_COLUMNS)

    obj = pickleio(path=dbpath_cleaned_log, mode="load")

    if isinstance(obj, pd.DataFrame):
        js = obj.copy()
        for col in JOB_STATE_COLUMNS:
            if col not in js.columns:
                js[col] = pd.NA
        return js[JOB_STATE_COLUMNS]

    if isinstance(obj, (set, list, tuple)):
        return pd.DataFrame({"file": list(map(str, obj))}, columns=JOB_STATE_COLUMNS)

    return pd.DataFrame(columns=JOB_STATE_COLUMNS)


def _save_job_state(job_state: pd.DataFrame) -> None:
    for col in JOB_STATE_COLUMNS:
        if col not in job_state.columns:
            job_state[col] = pd.NA
    job_state = job_state[JOB_STATE_COLUMNS]
    pickleio(path=dbpath_cleaned_log, data=job_state, mode="save")


def _upsert_job_state_row(
    job_state: pd.DataFrame,
    *,
    file: str,
    path: str,
    dir_name: str,
    date_key: Optional[str],
    item: Optional[str],
    state: Dict[str, Any],
) -> pd.DataFrame:
    if job_state.empty:
        idx = pd.Series([], dtype=bool)
    else:
        idx = (job_state["path"] == path)

    if not idx.any():
        row = {col: pd.NA for col in JOB_STATE_COLUMNS}
        row.update({"file": file, "path": path, "dir": dir_name, "date": date_key, "item": item})
        row.update(state)
        row_df = pd.DataFrame([row], columns=JOB_STATE_COLUMNS)

        if job_state.empty:
            job_state = row_df
        else:
            # 避免 FutureWarning：把全 NA row 過濾掉（雖然這裡通常不會）
            row_df = row_df.dropna(axis=1, how="all")
            job_state = pd.concat([job_state, row_df], ignore_index=True)

    else:
        for k, v in state.items():
            if k in job_state.columns:
                job_state.loc[idx, k] = v

        job_state.loc[idx, "file"] = file
        job_state.loc[idx, "path"] = path
        job_state.loc[idx, "dir"] = dir_name
        if date_key is not None:
            job_state.loc[idx, "date"] = date_key
        if item is not None:
            job_state.loc[idx, "item"] = item

    return job_state


def key_extract(dic: dict) -> list[dict]:
    if not isinstance(dic, dict):
        raise TypeError(f"key_extract() expects dict, got {type(dic).__name__}")

    out: list[dict] = []

    def _listify(x):
        return x if isinstance(x, (list, tuple)) else [x]

    def _find_first_key(container: dict, aliases: list[str]) -> tuple[str, bool]:
        for k in aliases:
            if k in container:
                return k, True
        return "", False

    def _extract_from_container(container: dict) -> list[dict]:
        dicts: list[dict] = []
        for step, set_i in key_set.items():
            if not isinstance(set_i, dict):
                continue

            if step == "main1":
                cnt = 0
                while True:
                    curr: dict = {}
                    for key_name, alias_list in set_i.items():
                        aliases = _listify(alias_list)
                        candidates = [a if cnt == 0 else f"{a}{cnt}" for a in aliases]
                        hit_key, ok = _find_first_key(container, candidates)
                        if ok:
                            curr[key_name] = container[hit_key]

                    if curr:
                        dicts.append(curr)
                    elif cnt > 1:
                        break

                    cnt += 1

            else:
                curr = {}
                for key_name, alias_list in set_i.items():
                    aliases = _listify(alias_list)
                    hit_key, ok = _find_first_key(container, aliases)
                    if ok:
                        curr[key_name] = container[hit_key]
                if curr:
                    dicts.append(curr)

        return dicts

    out.extend(_extract_from_container(dic))

    tables = dic.get("tables")
    if isinstance(tables, list):
        for t in tables:
            if isinstance(t, dict):
                out.extend(_extract_from_container(t))

    return out


def frameup_safe(d: Dict[str, Any]) -> pd.DataFrame:
    fields = list(d.get("fields", []))
    rows = list(d.get("data", []))
    if not fields or not rows:
        raise DataCleanError("frameup_safe：缺少 fields 或 data")
    trimmed = [r[: len(fields)] for r in rows]
    df = pd.DataFrame(trimmed, columns=_normalize_cols(fields))
    return df


def data_cleaned_groups(d: Dict[str, Any], span_cfg: Dict[str, Any]) -> pd.DataFrame:
    fields = list(d.get("fields", []))
    rows = list(d.get("data", []))
    if not fields or not rows:
        raise DataCleanError("data_cleaned_groups：缺少 fields 或 data")
    groups = span_cfg.get("groups")
    if not groups:
        raise DataCleanError("data_cleaned_groups：span_cfg 缺少 groups")
    start = 0
    col_names: List[str] = []
    for g in groups:
        size = int(g.get("size", 0))
        prefix = str(g.get("prefix", ""))
        if size <= 0:
            raise DataCleanError("span group size 非正數", value=size)
        end = start + size
        seg_fields = fields[start:end]
        seg_cols = [f"{prefix}{c}" for c in seg_fields]
        col_names.extend(seg_cols)
        start = end
    total = len(col_names)
    trimmed = [(r[:total] + [None] * max(0, total - len(r))) for r in rows]
    df = pd.DataFrame(trimmed, columns=_normalize_cols(col_names))
    return df


def finalize_dataframe(
    df: pd.DataFrame,
    *,
    item: str,
    subitem: str,
    date_key: str,
) -> pd.DataFrame:
    df = df.drop(columns=dropcol, errors="ignore")

    rename_cfg = transtonew_col.get(item, {}).get(subitem, {})
    if rename_cfg:
        df = df.rename(columns=rename_cfg)

    num_cfg = numericol.get(item, {}).get(subitem, {})
    df = safe_numeric_convert(df, num_cfg)

    if "date" not in df.columns:
        df.insert(0, "date", date_key)

    date_cfg = datecol.get(item, {}).get(subitem, ["date"])
    try:
        df = stringtodate(df, datecol=date_cfg, mode=4)
    except Exception as e:
        raise DataCleanError(
            "日期欄位轉換失敗",
            item=item, subitem=subitem, column=date_cfg,
            value_type=type(df).__name__,
            hint="請補充 stringtodate 規則或前置清理邏輯",
        ) from e

    front = [c for c in ["date", "代號", "名稱"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    return df


def _db_path_for_item(item: str) -> str:
    _ensure_dir(dbpath_cleaned)
    return join(dbpath_cleaned, f"{item}")


# =========================
# ✅ 一次到位：欄名唯一性檢查（fail-fast）
# =========================
def _assert_unique_columns(
    df: pd.DataFrame,
    *,
    file: Optional[str],
    item: str,
    subitem: str,
    table_name: str,
    stage: str,
) -> None:
    if df is None:
        return

    cols = pd.Index([str(c) for c in df.columns])
    if cols.is_unique:
        return

    dup = cols[cols.duplicated()].tolist()
    # 次數（避免印爆，只列前 60）
    vc = pd.Series(list(cols)).value_counts()
    uniq_dup = list(dict.fromkeys(dup))
    dup_counts = {k: int(vc[k]) for k in uniq_dup[:60]}

    raise DataCleanError(
        f"欄名重複（不允許）→ {stage} 立刻中止",
        file=file or "UNKNOWN",
        item=item,
        subitem=subitem,
        value={
            "table_name": table_name,
            "dup_cols": uniq_dup[:60],
            "dup_counts": dup_counts,
            "n_cols": int(len(cols)),
        },
        hint="這通常是 fields_span 沒套到（fallback frameup_safe）或原始表頭結構變了。請補 fields_span/groups 或調整 subtitle/欄位解析。",
    )


@dataclass
class _PendingWrite:
    db_path: Path
    item: str
    subitem: str
    table_name: str
    pk: List[str]
    partition_cols: Optional[List[str]]
    convert_mode: str
    is_me_freq: bool
    partition_by_date: bool
    bucket_mode: str
    dfs: List[pd.DataFrame] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # 每個 df 對應的 source 檔名


_DB_WRITE_BUFFER: Dict[Tuple[str, str], _PendingWrite] = {}


def _enqueue_db_write(
    df: pd.DataFrame,
    *,
    db_path: str,
    item: str,
    subitem: str,
    table_name: str,
    pk: List[str],
    partition_cols: Optional[List[str]],
    convert_mode: str,
    is_me_freq: bool,
    partition_by_date: bool,
    bucket_mode: str,
    source_file: Optional[str] = None,
) -> None:
    # ✅ fail-fast：concat 以前就擋掉（你要的「立刻中止」）
    _assert_unique_columns(
        df,
        file=source_file,
        item=item,
        subitem=subitem,
        table_name=table_name,
        stage="enqueue",
    )

    key = (str(db_path), table_name)

    if key not in _DB_WRITE_BUFFER:
        _DB_WRITE_BUFFER[key] = _PendingWrite(
            db_path=Path(db_path),
            item=item,
            subitem=subitem,
            table_name=table_name,
            pk=list(pk) if pk else [],
            partition_cols=list(partition_cols) if partition_cols else None,
            convert_mode=convert_mode,
            is_me_freq=is_me_freq,
            partition_by_date=True if partition_by_date else False,
            bucket_mode=bucket_mode,
        )

    pending = _DB_WRITE_BUFFER[key]
    pending.dfs.append(df.copy())
    pending.sources.append(str(source_file) if source_file else "UNKNOWN")


def _drop_dup_with_report(
    df: pd.DataFrame,
    pk: List[str],
    *,
    item: str,
    subitem: str,
    stage: str,
) -> pd.DataFrame:
    if df is None or df.empty or not pk:
        return df
    if not all(c in df.columns for c in pk):
        return df

    before = len(df)
    out = df.drop_duplicates(subset=pk, keep="last").copy()
    removed = before - len(out)

    if removed > 0:
        logger.info(
            "%s 去重：item=%s subitem=%s pk=%s removed=%d (rows %d -> %d)",
            stage, item, subitem, pk, removed, before, len(out)
        )
    return out


def _upsert_merge_existing(
    dbi: DBPkl,
    df_new: pd.DataFrame,
    pk: List[str],
    *,
    item: str,
    subitem: str,
) -> pd.DataFrame:
    if df_new is None or df_new.empty:
        return df_new
    if not pk or not all(c in df_new.columns for c in pk):
        return df_new

    try:
        existing = dbi.load_db()
    except FileNotFoundError:
        existing = None
    except Exception as e:
        logger.warning(
            "讀取既有資料失敗（將退化為直接寫入本批）：item=%s subitem=%s err=%s",
            item, subitem, e
        )
        existing = None

    if existing is None or existing.empty:
        return df_new

    merged = pd.concat([existing, df_new], ignore_index=True)

    before = len(merged)
    merged = merged.drop_duplicates(subset=pk, keep="last").copy()
    removed = before - len(merged)

    if removed > 0:
        logger.info(
            "UPSERT 去重（existing + new）：item=%s subitem=%s pk=%s removed=%d (rows %d -> %d) | exist=%d new=%d",
            item, subitem, pk, removed, before, len(merged), len(existing), len(df_new)
        )

    return merged


def _flush_db_write_buffer() -> None:
    global DEBUG_LAST_DF, DEBUG_LAST_CONTEXT
    global UPDATE_OLD_NON_DAILY

    if not _DB_WRITE_BUFFER:
        return

    for _, pending in list(_DB_WRITE_BUFFER.items()):
        if not pending.dfs:
            continue

        # ✅ 二次保險：理論上 enqueue 已擋掉，但保留避免有人繞過 _enqueue_db_write
        for i, df in enumerate(pending.dfs):
            _assert_unique_columns(
                df,
                file=pending.sources[i] if i < len(pending.sources) else "UNKNOWN",
                item=pending.item,
                subitem=pending.subitem,
                table_name=pending.table_name,
                stage="flush 前（buffer 內）",
            )

        try:
            dfs = []
            srcs = []
            for i, df in enumerate(pending.dfs):
                if df is None or df.empty:
                    continue
                dfs.append(df)
                if i < len(pending.sources):
                    srcs.append(pending.sources[i])

            pending.dfs = dfs
            pending.sources = srcs

            if not pending.dfs:
                continue

            df_new = pd.concat(pending.dfs, ignore_index=True, sort=False)

        except Exception as e:
            bad = []
            for i, df in enumerate(pending.dfs):
                cols = pd.Index([str(c) for c in df.columns])
                if not cols.is_unique:
                    dup_cols = cols[cols.duplicated()].tolist()
                    bad.append((i, pending.sources[i] if i < len(pending.sources) else "UNKNOWN", dup_cols[:60]))

            logger.error(
                "concat 失敗 | item=%s table=%s | bad_dfs=%s | err=%s",
                pending.item, pending.table_name, bad, e
            )
            raise

        dbi = DBPkl(
            str(pending.db_path),
            pending.table_name,
            logical_table_name=pending.subitem,
        )

        pk = pending.pk if pending.pk else None
        pk_list: List[str] = list(pk) if pk else []

        if pk_list:
            df_new = _drop_dup_with_report(
                df_new, pk_list,
                item=pending.item,
                subitem=pending.subitem,
                stage="flush 前（同批）"
            )

        non_daily = _is_non_daily_freq_item(pending.item)

        existing = None
        if non_daily and pk_list and all(c in df_new.columns for c in pk_list):
            try:
                existing = dbi.load_db()
            except FileNotFoundError:
                existing = None
            except Exception as e:
                logger.warning(
                    "讀取既有資料失敗（非日頻策略將退化為直接寫入）：item=%s subitem=%s err=%s",
                    pending.item, pending.subitem, e
                )
                existing = None

        if non_daily and existing is not None and not existing.empty and pk_list and all(c in existing.columns for c in pk_list):
            if UPDATE_OLD_NON_DAILY:
                before_exist = len(existing)
                before_new = len(df_new)

                merged = pd.concat([existing, df_new], ignore_index=True)
                merged = merged.drop_duplicates(subset=pk_list, keep="last").copy()

                logger.info(
                    "非日頻回補/覆寫舊資料（UPSERT）：item=%s subitem=%s exist=%d new=%d merged=%d pk=%s",
                    pending.item, pending.subitem, before_exist, before_new, len(merged), pk_list
                )
                df_new = merged
            else:
                exist_keys = existing[pk_list].drop_duplicates()
                before_new = len(df_new)

                tmp = df_new.merge(exist_keys, on=pk_list, how="left", indicator=True)
                df_new = tmp[tmp["_merge"] == "left_only"].drop(columns=["_merge"]).copy()

                removed = before_new - len(df_new)
                logger.info(
                    "非日頻不回補：移除已存在 PK：item=%s subitem=%s removed=%d remain=%d pk=%s",
                    pending.item, pending.subitem, removed, len(df_new), pk_list
                )

                if df_new.empty:
                    logger.info(
                        "非日頻不回補：本批全部 PK 已存在 DB，略過寫入。item=%s subitem=%s",
                        pending.item, pending.subitem
                    )
                    continue

        need_upsert = True
        if non_daily and (not UPDATE_OLD_NON_DAILY):
            need_upsert = False

        if need_upsert and pk_list:
            df_new = _upsert_merge_existing(
                dbi, df_new, pk_list,
                item=pending.item,
                subitem=pending.subitem
            )

        try:
            if pending.partition_cols:
                dbi.write_partition(
                    df_new,
                    convert_mode=pending.convert_mode,
                    partition_cols=pending.partition_cols,
                    primary_key=pk_list if pk_list else None,
                )
            else:
                dbi.write_db(
                    df_new,
                    convert_mode=pending.convert_mode,
                    primary_key=pk_list if pk_list else None,
                )

        except Exception as e:
            DEBUG_LAST_DF = df_new
            conflict = getattr(dbi, "schema_conflict", None)
            try:
                dtypes = df_new.dtypes.astype(str).to_dict()
            except Exception:
                dtypes = {}

            DEBUG_LAST_CONTEXT = {
                "item": pending.item,
                "subitem": pending.table_name,
                "db_path": str(pending.db_path),
                "pk": pending.pk,
                "convert_mode": pending.convert_mode,
                "conflict": conflict,
                "exception_type": type(e).__name__,
                "exception_str": str(e),
                "columns": list(df_new.columns),
                "shape": tuple(df_new.shape),
                "head": df_new.head(5),
                "dtypes": dtypes,
            }
            if conflict:
                logger.debug(f"[DB schema conflict] {conflict}")
            raise

    _DB_WRITE_BUFFER.clear()


def _write_to_db(
    df: pd.DataFrame,
    convert_mode: str = "coerce",
    *,
    item: str,
    subitem: str,
    bucket_mode: str = "all",
    source_file: str,
) -> None:
    pk: List[str] = []
    if "代號" in df.columns and "date" in df.columns:
        pk = ["代號", "date"]
    elif "名稱" in df.columns and "date" in df.columns:
        pk = ["名稱", "date"]
    elif "date" in df.columns:
        pk = ["date"]

    db_path = _db_path_for_item(item)

    partition_by_date = "date" in df.columns and _is_partition_by_date_item(item)
    is_me_freq = _is_me_freq_item(item)

    logger.debug(
        "寫入 DB（buffer 模式）：%s 表=%s PK=%s partition_by_date=%s bucket_mode=%s is_me_freq=%s",
        db_path,
        subitem,
        pk,
        partition_by_date,
        bucket_mode,
        is_me_freq,
    )

    if partition_by_date and bucket_mode.lower() != "all":
        bucket_key = _make_bucket_key(df["date"], bucket_mode)

        for b, df_chunk in df.groupby(bucket_key):
            table_name = f"{subitem}__{b}"

            logger.debug(
                "enqueue 分桶表至 buffer：%s/%s（bucket=%s, rows=%d）",
                db_path,
                table_name,
                b,
                len(df_chunk),
            )

            df_chunk = _repair_duplicate_columns(
                df_chunk,
                file_name=source_file,
                item=item,
                subitem=subitem,
                table_name=table_name,
            )

            _enqueue_db_write(
                df_chunk,
                db_path=db_path,
                item=item,
                subitem=subitem,
                table_name=table_name,
                pk=pk,
                partition_cols=["date"],
                convert_mode=convert_mode,
                is_me_freq=is_me_freq,
                partition_by_date=True,
                bucket_mode=bucket_mode,
                source_file=source_file,   # ✅ FIX：bucket 分支也要傳，否則 UNKNOWN
            )

        return

    table_name = subitem
    partition_cols = ["date"] if partition_by_date else None

    logger.debug(
        "enqueue 單一表至 buffer：%s/%s（rows=%d, partition_cols=%s）",
        db_path,
        table_name,
        len(df),
        partition_cols,
    )

    df = _repair_duplicate_columns(
        df,
        file_name=source_file,
        item=item,
        subitem=subitem,
        table_name=table_name,
    )

    _enqueue_db_write(
        df,
        db_path=db_path,
        item=item,
        subitem=subitem,
        table_name=table_name,
        pk=pk,
        partition_cols=partition_cols,
        convert_mode=convert_mode,
        is_me_freq=is_me_freq,
        partition_by_date=partition_by_date,
        bucket_mode=bucket_mode,
        source_file=source_file,
    )


# ---- 清洗一個檔案（主流程子步驟） ----
def _process_one_file(
    file_path: str,
    *,
    bucket_mode: str = "all",
    update_old_non_daily: bool = False,
) -> Tuple[str, str, str]:

    file_name = basename(file_path)
    parentdir = Path(file_path).parent.name
    logger.info(f"處理檔案：{file_name}（類別={parentdir}）")

    t0 = time.perf_counter()

    raw = pickleio(path=file_path, mode="load")
    if not isinstance(raw, dict):
        raise DataCleanError("原始 pkl 非 dict 結構", file=file_name)

    t1 = time.perf_counter()

    try:
        date_key = raw.get("crawlerdic", {}).get("payload", {}).get("date")
    except Exception as e:
        raise DataCleanError("無法取得 crawler 日期", file=file_name, item=parentdir, value=raw.get("crawlerdic")) from e

    subtitle_from_crawler = raw.get("crawlerdic", {}).get("subtitle")
    if isinstance(subtitle_from_crawler, list) and subtitle_from_crawler:
        subtitle_allowed = [colname_dic.get(x, x) for x in subtitle_from_crawler]
    else:
        subtitle_allowed = [colname_dic.get(x, x) for x in (collection.get(parentdir, {}).get("subtitle") or [parentdir])]

    if not subtitle_allowed:
        raise RuntimeError(
            f"subtitle_allowed 為空，無法判定清理目標；"
            f"file={file_name}, item={parentdir}. "
            f"請檢查 crawlerdic.subtitle 或 config.collection['{parentdir}']['subtitle']"
        )

    sub_tables = key_extract(raw)

    if not sub_tables:
        legacy_subs = _extract_legacy_subtables(raw)
        if legacy_subs:
            logger.warning(
                "未依 config 找到子表，改用 legacy fieldsN/dataN 掃描方式：file=%s, item=%s, legacy_subtitles=%r",
                file_name,
                parentdir,
                [s.get("title") for s in legacy_subs],
            )
            sub_tables = legacy_subs
        else:
            raise DataCleanError(
                "未找到任何可清理的子表",
                file=file_name,
                item=parentdir,
                value=list(raw.keys()),
            )

    clean_time_total = 0.0
    write_time_total = 0.0

    for _, d in enumerate(sub_tables, 1):
        title = d.get("title")
        fields = d.get("fields")
        data = d.get("data")
        if not fields or not data:
            logger.debug(f"略過子表（無資料）：title={title!r}")
            continue

        subitem_raw = _match_subtitle_from_conf_by_contains(
            item=parentdir,
            raw_title=title,
            file_name=file_name,
        )

        subitem = colname_dic.get(subitem_raw, subitem_raw)

        logger.debug(
            "清理子表：%s（原 title=%r, matched subtitle=%r）",
            subitem,
            title,
            subitem_raw,
        )

        t_clean_start = time.perf_counter()
        try:
            span_cfg = _get_span_cfg(parentdir, subitem)

            if span_cfg is not None:
                df0 = data_cleaned_groups({"fields": fields, "data": data}, span_cfg)
            else:
                df0 = frameup_safe({"fields": fields, "data": data})

            df1 = finalize_dataframe(df0, item=parentdir, subitem=subitem, date_key=date_key)

        except DataCleanError:
            raise
        except Exception as e:
            raise DataCleanError(
                "子表清理失敗",
                file=file_name, item=parentdir, subitem=subitem, date=date_key,
                hint="請檢查 fields_span/dropcol/transtonew_col/numericol/datecol 與原始資料是否一致",
            ) from e

        t_clean_end = time.perf_counter()
        clean_time_total += (t_clean_end - t_clean_start)

        t_write_start = time.perf_counter()

        df1 = _apply_non_daily_policy(
            df1,
            item=parentdir,
            subitem=subitem,
            file_name=file_name,
            update_old_non_daily=update_old_non_daily,
        )

        _write_to_db(
            df1,
            item=parentdir,
            subitem=subitem,
            bucket_mode=bucket_mode,
            source_file=file_name,
        )
        t_write_end = time.perf_counter()
        write_time_total += (t_write_end - t_write_start)

    t_end = time.perf_counter()

    read_time = t1 - t0
    total_time = t_end - t0

    logger.info(
        "檔案耗時統計：%s | read=%.2fs, clean=%.2fs, write=%.2fs, total=%.2fs",
        file_name,
        read_time,
        clean_time_total,
        write_time_total,
        total_time,
    )

    return date_key, parentdir, file_name


# ---- 清洗流程（可被 import 呼叫） ----
def _process_twse_data_impl(
    cols: Optional[List[str]] = None,
    max_files_per_run: Optional[int] = None,
    bucket_mode: str = "all",
    update_old_non_daily: bool = False,
) -> int:
    _ensure_dir(str(dbpath_cleaned))

    job_state = _load_job_state()
    files_df = _list_source_pickles(dbpath_source)

    if cols:
        files_df = files_df[files_df["dir"].isin(cols)].copy()

    total_files = len(files_df)
    if files_df.empty:
        logger.info("找不到任何待處理的 source 檔案。")
        return 0

    remaining_est = _estimate_remaining_by_job_state(files_df, job_state)
    logger.info("待檢查檔案數：%d | remaining_est=%d", total_files, remaining_est)

    if remaining_est == 0:
        logger.info("remaining_est=0，進行一次精準確認（避免漏掉 source 變更）...")
        if not _has_any_pending_exact(files_df, job_state):
            logger.info("確認無待處理檔案 → 本輪不進處理流程。")
            return 0
        logger.info("精準確認：仍有待處理檔案（可能是 source 變更）→ 進入處理流程。")

    processed = 0
    skipped_unchanged = 0
    start_time = datetime.now()

    try:
        for scanned_idx, (_, row) in enumerate(files_df.iterrows(), start=1):
            if max_files_per_run is not None and processed >= max_files_per_run:
                _overwrite_line_end()
                logger.info(
                    "已達本輪處理上限 %d 檔，本輪提前結束（實際處理 %d 檔，掃描到第 %d 檔 / 總檔數 %d）。",
                    max_files_per_run,
                    processed,
                    scanned_idx - 1,
                    total_files,
                )
                break

            file_path = row["path"]
            file_name = row["file"]
            dir_name = row["dir"]

            state_now = _calc_file_state(file_path)
            fp_now = state_now["hash"]
            mtime_now = state_now["source_mtime"]
            size_now = state_now["source_size"]

            rec_idx = (job_state["path"] == file_path) if not job_state.empty else pd.Series([], dtype=bool)
            rec = job_state.loc[rec_idx].iloc[0] if rec_idx.any() else None

            if rec is not None:
                rec_status = rec.get("status")
                rec_hash = rec.get("hash")
                rec_mtime = rec.get("source_mtime")

                if (
                    rec_status == "success"
                    and pd.notna(rec_hash)
                    and str(rec_hash) == str(fp_now)
                    and pd.notna(rec_mtime)
                    and pd.Timestamp(rec_mtime) == mtime_now
                ):
                    skipped_unchanged += 1
                    rem = max(remaining_est - processed, 0)
                    _overwrite_line(
                        f"[skip] scanned {scanned_idx}/{total_files} | processed {processed} | skipped {skipped_unchanged} | remaining_est {rem}"
                    )
                    continue

                if rec_status == "success" and (
                    str(rec_hash) != str(fp_now)
                    or (pd.notna(rec_mtime) and pd.Timestamp(rec_mtime) != mtime_now)
                ):
                    _overwrite_line_end()
                    logger.warning(
                        "偵測到 source 在上次成功清理後有變更，標記為 pending：file=%s, old_mtime=%s, new_mtime=%s",
                        file_name,
                        rec_mtime,
                        mtime_now,
                    )
                    job_state.loc[rec_idx, "status"] = "pending"

            job_state = _upsert_job_state_row(
                job_state,
                file=file_name,
                path=file_path,
                dir_name=dir_name,
                date_key=None,
                item=None,
                state={
                    "status": "pending",
                    "hash": fp_now,
                    "source_mtime": mtime_now,
                    "source_size": size_now,
                },
            )
            _save_job_state(job_state)

            _overwrite_line_end()

            try:
                date_key, item, cleaned_file_name = _process_one_file(
                    file_path,
                    bucket_mode=bucket_mode,
                    update_old_non_daily=update_old_non_daily,
                )
            except Exception as e:
                job_state = _upsert_job_state_row(
                    job_state,
                    file=file_name,
                    path=file_path,
                    dir_name=dir_name,
                    date_key=None,
                    item=None,
                    state={
                        "status": "failed",
                        "hash": fp_now,
                        "source_mtime": mtime_now,
                        "source_size": size_now,
                        "last_processed_at": pd.Timestamp.utcnow(),
                    },
                )
                _save_job_state(job_state)
                logger.error("處理發生錯誤：%s", e)
                raise

            job_state = _upsert_job_state_row(
                job_state,
                file=file_name,
                path=file_path,
                dir_name=dir_name,
                date_key=date_key,
                item=item,
                state={
                    "status": "success",
                    "hash": fp_now,
                    "source_mtime": mtime_now,
                    "source_size": size_now,
                    "last_processed_at": pd.Timestamp.utcnow(),
                },
            )
            _save_job_state(job_state)

            processed += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_sec = elapsed / processed if processed else 0.0
            rem = max(remaining_est - processed, 0)

            logger.info(
                "完成：%s（date=%s, item=%s）｜本輪已處理 %d 檔 / remaining_est %d｜掃描 %d/%d｜skip %d｜平均 %.1f 秒/檔。",
                cleaned_file_name,
                date_key,
                item,
                processed,
                rem,
                scanned_idx,
                total_files,
                skipped_unchanged,
                avg_sec,
            )

    finally:
        _overwrite_line_end()
        _flush_db_write_buffer()

    return processed


def process_twse_data(
    cols: Optional[List[str]] = None,
    *,
    storage_mode: str = "cloud",
    batch_size: Optional[int] = None,
    bucket_mode: str = "all",
    update_old_non_daily: bool = False,
) -> None:
    global dbpath_source, dbpath_cleaned, dbpath_cleaned_log
    global UPDATE_OLD_NON_DAILY
    UPDATE_OLD_NON_DAILY = bool(update_old_non_daily)

    logger.info(
        "非日頻回補旗標：update_old_non_daily=%s（freq!=D 時：True=upsert 覆寫舊資料；False=只取最新 date 且不覆寫）",
        UPDATE_OLD_NON_DAILY
    )

    storage_mode = (storage_mode or "cloud").lower()
    if storage_mode not in {"cloud", "cloud_staging", "local"}:
        raise ValueError(f"storage_mode 必須是 'cloud' / 'cloud_staging' / 'local'，目前為：{storage_mode!r}")

    logger.info(
        "process_twse_data 啟動：storage_mode=%s, bucket_mode=%s, dbpath_source(初始)=%s, dbpath_cleaned(初始)=%s",
        storage_mode,
        bucket_mode,
        dbpath_source,
        dbpath_cleaned,
    )

    orig_source = dbpath_source
    orig_cleaned = dbpath_cleaned
    orig_cleaned_log = dbpath_cleaned_log

    # ---------- A：完全本機模式 ----------
    if storage_mode == "local":
        logger.info("LOCAL 模式啟動，先清空本機 db_local_root：%s", LOCAL_DB_ROOT)
        _clear_local_paths()

        dbpath_source = LOCAL_DBPATH_SOURCE
        dbpath_cleaned = LOCAL_DBPATH_CLEANED
        dbpath_cleaned_log = LOCAL_DBPATH_CLEANED_LOG

        _ensure_dir(str(dbpath_source))
        _ensure_dir(str(dbpath_cleaned))

        logger.info("進入 LOCAL 模式：source=%s, cleaned=%s", dbpath_source, dbpath_cleaned)

        try:
            _process_twse_data_impl(
                cols,
                bucket_mode=bucket_mode,
                max_files_per_run=batch_size,
                update_old_non_daily=update_old_non_daily,
            )
        finally:
            dbpath_source = orig_source
            dbpath_cleaned = orig_cleaned
            dbpath_cleaned_log = orig_cleaned_log

        return

    # ---------- B/C：以雲端為主 ----------
    dbpath_source = CLOUD_DBPATH_SOURCE
    dbpath_cleaned = CLOUD_DBPATH_CLEANED
    dbpath_cleaned_log = CLOUD_DBPATH_CLEANED_LOG

    # B-1：純雲端
    if storage_mode == "cloud":
        logger.info("進入 CLOUD (no staging) 模式：source=%s, cleaned=%s", dbpath_source, dbpath_cleaned)
        try:
            _process_twse_data_impl(
                cols,
                bucket_mode=bucket_mode,
                max_files_per_run=batch_size,
                update_old_non_daily=update_old_non_daily,
            )
        finally:
            dbpath_source = orig_source
            dbpath_cleaned = orig_cleaned
            dbpath_cleaned_log = orig_cleaned_log
        return

    # B-2：雲端 + 本機 staging
    if batch_size is None:
        batch_size = 10_000_000

    target_cleaned: Path = CLOUD_DBPATH_CLEANED
    staging_root: Path = LOCAL_DB_ROOT

    logger.info("cloud_staging 啟動前，先清理本機 staging 目錄（root=%s）", staging_root)
    _clear_staging_dirs(staging_root)

    batch_no = 0
    synced_batches = 0

    while True:
        batch_no += 1

        try:
            tmp_job_state = _load_job_state()
            tmp_files_df = _list_source_pickles(dbpath_source)
            if cols:
                tmp_files_df = tmp_files_df[tmp_files_df["dir"].isin(cols)].copy()
            remaining_est = _estimate_remaining_by_job_state(tmp_files_df, tmp_job_state)
        except Exception as e:
            logger.warning("預檢 remaining_est 失敗（保守起見照常進 staging）：err=%s", e)
            remaining_est = 1
            tmp_files_df = pd.DataFrame()
            tmp_job_state = pd.DataFrame(columns=JOB_STATE_COLUMNS)

        if remaining_est == 0:
            logger.info("staging batch %d 預檢 remaining_est=0，進行一次精準確認...", batch_no)
            if not _has_any_pending_exact(tmp_files_df, tmp_job_state):
                logger.info("確認無待處理檔案 → 不進 staging，結束迴圈。已同步批次=%d", synced_batches)
                break
            logger.info("精準確認：仍有待處理檔案（可能是 source 變更）→ 繼續進 staging。")

        remaining_batches_est = math.ceil(remaining_est / batch_size) if remaining_est > 0 else 0

        logger.info(
            "===== 開始 staging batch %d | batch_size=%d | remaining_est=%d | remaining_batches_est=%d | synced_batches=%d =====",
            batch_no,
            batch_size,
            remaining_est,
            remaining_batches_est,
            synced_batches,
        )

        with staging_path(target_cleaned, enable=True, staging_root=staging_root) as local_cleaned:
            try:
                dbpath_cleaned = local_cleaned
                dbpath_cleaned_log = local_cleaned / "log.pkl"

                logger.info("cloud_staging 模式：本輪在本機 cleaned=%s 上處理", dbpath_cleaned)

                processed = _process_twse_data_impl(
                    cols,
                    bucket_mode=bucket_mode,
                    max_files_per_run=batch_size,
                    update_old_non_daily=update_old_non_daily,
                )
            finally:
                dbpath_cleaned = CLOUD_DBPATH_CLEANED
                dbpath_cleaned_log = CLOUD_DBPATH_CLEANED_LOG

        synced_batches += 1

        if processed == 0:
            logger.info("本輪 processed=0（理論上不應發生，因為前面已擋掉 remaining_est=0），staging 迴圈結束。")
            break

        logger.info("===== staging batch %d 完成，本輪處理 %d 個檔案 | synced_batches=%d =====", batch_no, processed, synced_batches)

    dbpath_source = orig_source
    dbpath_cleaned = orig_cleaned
    dbpath_cleaned_log = orig_cleaned_log


# ---- CLI ----
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TWSE 資料清理器（合併版）")
    p.add_argument("--col", nargs="*", help="指定要清理的類別（資料夾名/collection key），預設全清")
    p.add_argument(
        "--storage-mode",
        type=str,
        default="cloud",
        choices=["cloud", "cloud_staging", "local"],
        help="cloud=直接用雲端；cloud_staging=雲端+本機暫存；local=完全只用本機 db_local_root。",
    )
    p.add_argument("--batch-size", type=int, default=None, help="每一輪 staging 要處理的最大檔案數")
    p.add_argument(
        "--bucket-mode",
        choices=["all", "year", "quarter", "month", "day"],
        default="all",
        help="日期分桶模式：all=整檔一個表、year=每年一表、quarter=每季一表、month=每月一表、day=每日一表",
    )
    p.add_argument(
        "--update-old-non-daily",
        action="store_true",
        help="非日頻（freq!=D）允許回補/覆寫舊日期資料（UPSERT）。預設關閉：只取最新 date 且不覆寫既有 PK。",
    )

    args, unknown = p.parse_known_args(argv)
    if unknown:
        logger.debug("忽略未識別參數：%s", unknown)
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    process_twse_data(
        cols=args.col,
        storage_mode=args.storage_mode,
        batch_size=args.batch_size,
        bucket_mode=args.bucket_mode,
        update_old_non_daily=args.update_old_non_daily,
    )


if __name__ == "__main__":
    # main()

    process_twse_data(
        cols=None,
        storage_mode="cloud_staging",
        batch_size=500,
        bucket_mode="year",
        update_old_non_daily=False,
    )
