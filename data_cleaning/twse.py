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

依賴（請對齊你專案實際模組）
- StevenTricks.file_utils: pickleio, PathWalk_df
- StevenTricks.convert_utils: safe_replace, safe_numeric_convert, changetype_stringtodate
- StevenTricks.dict_utils: keyinstr
- StevenTricks.internal_db: tosql_df
- config.conf: collection, fields_span, dropcol, key_set
- config.col_rename: colname_dic, transtonew_col
- config.col_format: numericol, datecol
- config.paths: dbpath_source, dbpath_clean(或fallback), dbpath_clean_log(或fallback)

用法（CLI）
    清理所有類別：
        python -m data_cleaning.twse
    只清理指定類別（資料夾名/collection key）：
        python -m data_cleaning.twse --col 每日收盤行情 信用交易統計

用法（import）
    from data_cleaning.twse import process_twse_data, clean_one_dataframe
    process_twse_data(["每日收盤行情"])
    # 或單測：
    df2 = clean_one_dataframe(df_raw, item="每日收盤行情", subitem="個股", date="2025-08-10")
"""

from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from os import makedirs
from os import walk
from os.path import dirname, exists, join, splitext, basename
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# ---- StevenTricks 與 config ----
from StevenTricks.file_utils import pickleio
try:
    # 新版若有 PathWalk_df，用它；沒有就走 fallback
    from StevenTricks.file_utils import PathWalk_df as _PathWalk_df  # type: ignore
except Exception:
    _PathWalk_df = None  # fallback 使用 os.walk

from StevenTricks.convert_utils import safe_replace, safe_numeric_convert, changetype_stringtodate
from StevenTricks.dict_utils import keyinstr
from StevenTricks import internal_db as db

from config.conf import collection, fields_span, dropcol, key_set
from config.col_rename import colname_dic, transtonew_col
from config.col_format import numericol, datecol

# 路徑：若新清洗路徑未定義，fallback 至 source 同層的 cleaned
try:
    from config.paths import dbpath_source  # 原始 pkl 的根目錄（crawler 產物）
except Exception as e:
    raise ImportError("config.paths 需要提供 dbpath_source（TWSE 原始資料根目錄）") from e

try:
    from config.paths import dbpath_clean  # 清洗後 SQLite 目錄
except Exception:
    dbpath_clean = join(dirname(dbpath_source), "cleaned")

try:
    from config.paths import dbpath_clean_log  # 清洗完成 log（pickle）
except Exception:
    dbpath_clean_log = join(dbpath_clean, "clean_log.pkl")


# ---- Logging：預設 DEBUG ----
_root = logging.getLogger()
if not _root.handlers:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ---- 自訂錯誤類，讓錯誤情境更清楚 ----
class DataCleanError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        file: Optional[str] = None,
        item: Optional[str] = None,
        subitem: Optional[str] = None,
        column: Optional[str] = None,
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
        if column: ctx.append(f"column={column}")
        if value is not None: ctx.append(f"value={repr(value)}")
        if value_type: ctx.append(f"value_type={value_type}")
        if date: ctx.append(f"date={date}")
        if hint: ctx.append(f"hint={hint}")
        if ctx:
            parts.append(" | " + ", ".join(ctx))
        super().__init__("".join(parts))


# ---- 小工具 ----
def _ensure_dir(p: str) -> None:
    makedirs(p, exist_ok=True)


def _normalize_cols(cols: List[str]) -> List[str]:
    """欄名歸一：先 colname_dic 映射，再移除 HTML 斷行。"""
    mapped = [colname_dic.get(c, c) for c in cols]
    cleaned = [safe_replace(c, "</br>", "") for c in mapped]
    return cleaned


def _to_date_key(s: Any) -> str:
    """
    將 crawler 的 'YYYYMMDD'、或 'YYYY-MM-DD' 轉為 'YYYY-MM-DD'。
    若無法解析，丟 DataCleanError（立刻停）。
    """
    # 盡量寬鬆接受：純數字8碼 / 有連字號
    if isinstance(s, (int, float)) and not pd.isna(s):
        s = str(int(s))
    if isinstance(s, str):
        t = s.strip()
        if len(t) == 8 and t.isdigit():
            # YYYYMMDD
            key = f"{t[0:4]}-{t[4:6]}-{t[6:8]}"
        else:
            # 交給 pandas
            ts = pd.to_datetime(t, errors="coerce")
            if pd.isna(ts):
                raise DataCleanError("無法解析日期字串", value=s, value_type=type(s).__name__)
            key = ts.strftime("%Y-%m-%d")
        return key
    # 其它型別也交給 pandas
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise DataCleanError("無法解析日期（非字串型）", value=s, value_type=type(s).__name__)
    return ts.strftime("%Y-%m-%d")


def _list_source_pickles(root: str) -> pd.DataFrame:
    """
    列出 root 下所有 .pkl 檔；回傳 DataFrame 包含 columns: file, path, dir（上層資料夾名）
    優先用 StevenTricks.PathWalk_df，否則 fallback os.walk
    """
    if _PathWalk_df is not None:
        df = _PathWalk_df(root, [], ["log"], [".DS_Store"], [".pkl"])  # 依你的慣例
        # 期待有 'file', 'path', 'dir' 欄；若沒有就補
        need_cols = {"file", "path", "dir"}
        have = set(df.columns)
        if not need_cols.issubset(have):
            # 嘗試補齊
            if "path" not in df.columns:
                raise DataCleanError("PathWalk_df 缺少 path 欄")
            df["file"] = df["path"].map(lambda p: basename(p))
            df["dir"] = df["path"].map(lambda p: Path(p).parent.name)
        return df[["file", "path", "dir"]].copy()

    # fallback
    rows = []
    for dp, dnames, fnames in walk(root):
        # 排除 log 目錄
        if Path(dp).name.lower() == "log":
            continue
        for fn in fnames:
            if splitext(fn)[1].lower() == ".pkl":
                p = join(dp, fn)
                rows.append({"file": fn, "path": p, "dir": Path(dp).name})
    return pd.DataFrame(rows)


# ---- 解析 TWSE API 結構 → 子表 dict list ----
def key_extract(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    依 key_set 規則自 raw 結構中抽出子表字典。
    期望每個子表都至少帶有：title, fields, data
    """
    dict_list: List[Dict[str, Any]] = []

    # 1) 直屬 keys（常見 TWSE 回傳）
    for ks in key_set:  # e.g., [{"fields": "fields", "data":"data", "title":"title"}] 這類規格
        fields_key = ks.get("fields")
        data_key = ks.get("data")
        title_key = ks.get("title")
        if fields_key in raw and data_key in raw:
            entry = {
                "fields": raw.get(fields_key),
                "data": raw.get(data_key),
                "title": raw.get(title_key, None),
            }
            dict_list.append(entry)

    # 2) 若 raw 中還有 tables（多表），逐一處理
    tables = raw.get("tables")
    if isinstance(tables, list):
        for t in tables:
            for ks in key_set:
                fields_key = ks.get("fields")
                data_key = ks.get("data")
                title_key = ks.get("title")
                if isinstance(t, dict) and (fields_key in t and data_key in t):
                    entry = {
                        "fields": t.get(fields_key),
                        "data": t.get(data_key),
                        "title": t.get(title_key, None),
                    }
                    dict_list.append(entry)

    return dict_list


# ---- 兩種 DataFrame 組裝 ----
def frameup_safe(d: Dict[str, Any]) -> pd.DataFrame:
    """
    無群組欄位：直接以 d['fields'] 對齊 d['data']。
    若 data 的欄數 > fields，超出者丟棄（結構噪音），但不丟列。
    """
    fields = list(d.get("fields") or [])
    rows = list(d.get("data") or [])

    if not fields or not rows:
        raise DataCleanError("frameup_safe：缺少 fields 或 data")

    # 截斷每列到 len(fields)，避免野欄位
    trimmed = [r[: len(fields)] for r in rows]
    df = pd.DataFrame(trimmed, columns=_normalize_cols(fields))
    return df


def data_cleaned_groups(d: Dict[str, Any], span_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    有群組欄位（如 融資/融券）：
    span_cfg 例： {"groups": [{"prefix":"融資_", "size":5}, {"prefix":"融券_","size":5}], "index": ["日期","代號",...]}
    """
    fields = list(d.get("fields") or [])
    rows = list(d.get("data") or [])
    if not fields or not rows:
        raise DataCleanError("data_cleaned_groups：缺少 fields 或 data")

    groups = span_cfg.get("groups")
    if not groups:
        raise DataCleanError("data_cleaned_groups：span_cfg 缺少 groups")

    # 計算每群組欄位數，產生目標欄名
    df_list = []
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

    # 砍尾（若 fields 比群組總長還長），或不足（補空欄）→ 皆以「不刪列」為原則
    total = len(col_names)
    trimmed = [ (r[:total] + [None] * max(0, total - len(r))) for r in rows ]
    df = pd.DataFrame(trimmed, columns=_normalize_cols(col_names))
    return df


# ---- 最終欄位清理（drop/rename/numeric/date） ----
def finalize_dataframe(
    df: pd.DataFrame,
    *,
    item: str,
    subitem: str,
    date_key: str,
) -> pd.DataFrame:
    """
    依 config 規則產生最終清洗後 DataFrame；遇到未知欄位/型別問題直接丟 DataCleanError。
    """
    # 1) 移除不需要的欄（例如 漲跌(+/-)）
    drop_cfg = (dropcol.get(item) or {}).get(subitem) or []
    for c in drop_cfg:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 2) 欄名舊→新（細項規則）
    rename_cfg = (transtonew_col.get(item) or {}).get(subitem) or {}
    if rename_cfg:
        df = df.rename(columns=rename_cfg)

    # 3) 數值欄位轉換
    num_cfg = (numericol.get(item) or {}).get(subitem) or []
    for c in num_cfg:
        if c in df.columns:
            df[c] = safe_numeric_convert(df[c])  # 逗號、空白會被處理；非法轉 NaN（不丟列）

    # 4) 日期欄位轉換（若有指定 datecol）
    date_cfg = (datecol.get(item) or {}).get(subitem) or []
    for c in date_cfg:
        if c in df.columns:
            try:
                df[c] = changetype_stringtodate(df[c], mode=3)  # 你專案原本使用 mode=3（常見 ROC/多格式）
            except Exception as e:
                # 報出第一個壞值、型別
                bad = df[c].iloc[0]
                raise DataCleanError(
                    "日期欄位轉換失敗",
                    item=item, subitem=subitem, column=c,
                    value=bad, value_type=type(bad).__name__,
                    hint="請補充 changetype_stringtodate 規則或前置清理邏輯",
                ) from e

    # 5) 若沒有任何日期欄，補一個統一 'date'
    if "date" not in df.columns:
        df.insert(0, "date", pd.to_datetime(date_key, format="%Y-%m-%d"))

    # 6) 欄位順序微整：把常見鍵放前面
    front = [c for c in ["date", "代號"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    return df


# ---- 單一資料框清理（方便單測） ----
def clean_one_dataframe(
    df_raw: pd.DataFrame,
    *,
    item: str,
    subitem: str,
    date: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    對一個原始 DataFrame 進行最終規範化（不含 frameup/group 拆解步驟）
    - 適合你在 Notebook/REPL 針對某子表做單獨除錯
    """
    date_key = _to_date_key(date)
    df_raw = df_raw.copy()
    df_raw.columns = _normalize_cols(list(df_raw.columns))
    return finalize_dataframe(df_raw, item=item, subitem=subitem, date_key=date_key)


# ---- 寫入資料庫 ----
def _db_path_for_item(item: str) -> str:
    _ensure_dir(dbpath_clean)
    return join(dbpath_clean, f"{item}.db")


def _write_to_db(df: pd.DataFrame, *, item: str, subitem: str) -> None:
    """
    預設 PK 規則：
      - 同時有 '代號' 與 'date' → ['代號','date']
      - 僅有 'date' → ['date']
      - 否則不設 PK（由 tosql_df 處理）
    """
    pk: List[str] = []
    if "代號" in df.columns and "date" in df.columns:
        pk = ["代號", "date"]
    elif "date" in df.columns:
        pk = ["date"]
    db_path = _db_path_for_item(item)
    logger.debug(f"寫入 DB：{db_path} 表={subitem} PK={pk}")
    db.tosql_df(df, db_path, subitem, pk)  # 依 StevenTricks 目前介面


# ---- 清洗一個檔案（主流程子步驟） ----
def _process_one_file(file_path: str) -> Tuple[str, str, str]:
    """
    清洗單一 .pkl 檔。
    回傳：(date_key, item, file_name)
    """
    file_name = basename(file_path)
    parentdir = Path(file_path).parent.name  # 作為 item
    logger.info(f"處理檔案：{file_name}（類別={parentdir}）")

    raw = pickleio(path=file_path, mode="load")
    if not isinstance(raw, dict):
        raise DataCleanError("原始 pkl 非 dict 結構", file=file_name)

    # 特例：productlist 之類的靜態表，直接轉存 DB 或放過
    base, _ = splitext(file_name)
    if base.lower() == "productlist":
        # 你若需要寫入 DB，也可在此處理；此處先略過
        logger.info("偵測到 productlist.pkl：略過清洗流程（需另行定義）")
        return ("1970-01-01", parentdir, file_name)

    # 取 crawler 取得日
    try:
        date_raw = (((raw.get("crawlerdic") or {}).get("payload") or {}).get("date"))
        date_key = _to_date_key(date_raw)
    except Exception as e:
        raise DataCleanError("無法取得 crawler 日期", file=file_name, item=parentdir, value=raw.get("crawlerdic")) from e

    # 決定允許的子表（標準化後）
    # 以 crawler 的 subtitle 優先，否則取 config.collection[item]['subtitle']
    subtitle_from_crawler = (raw.get("crawlerdic") or {}).get("subtitle")
    if isinstance(subtitle_from_crawler, list) and subtitle_from_crawler:
        subtitle_allowed = [colname_dic.get(x, x) for x in subtitle_from_crawler]
    else:
        subtitle_allowed = [colname_dic.get(x, x) for x in (collection.get(parentdir, {}).get("subtitle") or [parentdir])]

    # 取所有子表（title, fields, data）
    sub_tables = key_extract(raw)
    if not sub_tables:
        raise DataCleanError("未找到任何可清理的子表", file=file_name, item=parentdir)

    for idx, d in enumerate(sub_tables, 1):
        title = d.get("title")
        fields = d.get("fields")
        data = d.get("data")
        if not fields or not data:
            logger.debug(f"略過子表（無資料）：title={title!r}")
            continue

        # 標準化子表名稱
        subitem = keyinstr(title, dic=colname_dic, lis=subtitle_allowed, default=str(title) if title is not None else parentdir)

        # 非預期子表：跳過（不是錯誤）
        if subitem not in subtitle_allowed:
            logger.debug(f"略過子表（不在允許清單）：title={title!r}, 標準名={subitem!r}")
            continue

        logger.debug(f"清理子表：{subitem}（原 title={title!r}）")

        # 組裝 DataFrame（群組 or 平面）
        try:
            if subitem in (fields_span.get(parentdir) or {}) or subitem in fields_span:
                # 支援兩種配置：以 item 為 key，或以 subitem 為 key
                span_cfg = (fields_span.get(parentdir, {}).get(subitem)) or (fields_span.get(subitem))
                if not span_cfg:
                    raise DataCleanError("找不到群組欄位設定", item=parentdir, subitem=subitem)
            else:
                span_cfg = None

            if span_cfg:
                df0 = data_cleaned_groups({"fields": fields, "data": data}, span_cfg)
            else:
                df0 = frameup_safe({"fields": fields, "data": data})

            # 最終規範化（drop/rename/numeric/date）
            df1 = finalize_dataframe(df0, item=parentdir, subitem=subitem, date_key=date_key)

        except DataCleanError:
            # 直接往外拋（你規定遇錯中斷）
            raise
        except Exception as e:
            # 包裝成 DataCleanError，附加更多上下文
            raise DataCleanError(
                "子表清理失敗",
                file=file_name, item=parentdir, subitem=subitem, date=date_key,
                hint="請檢查 fields_span/dropcol/transtonew_col/numericol/datecol 與原始資料是否一致",
            ) from e

        # 寫入 DB（每個子表一張表）
        _write_to_db(df1, item=parentdir, subitem=subitem)

    return (date_key, parentdir, file_name)


# ---- 清洗流程（可被 import 呼叫） ----
def process_twse_data(cols: Optional[List[str]] = None) -> None:
    """
    執行清洗流程。
    參數：
      cols: 要清的 collection 類別（資料夾名）；None 表示全部
    """
    _ensure_dir(dbpath_clean)
    # 載入或建立清洗 log（紀錄已處理檔名，避免重複清）
    if exists(dbpath_clean_log):
        clean_log = pickleio(path=dbpath_clean_log, mode="load")
        # 支援 DataFrame 或 set：都轉為 set of file names
        if isinstance(clean_log, pd.DataFrame):
            done = set([x for x in clean_log.values.ravel() if isinstance(x, str)])
        elif isinstance(clean_log, (set, list, tuple)):
            done = set(map(str, clean_log))
        else:
            done = set()
    else:
        clean_log = pd.DataFrame()
        done = set()

    # 列檔
    files_df = _list_source_pickles(dbpath_source)
    if cols:
        files_df = files_df[files_df["dir"].isin(cols)].copy()

    # 排除已做過的
    pending = files_df[~files_df["file"].isin(done)].copy().reset_index(drop=True)
    if pending.empty:
        logger.info("沒有需要清理的檔案。")
        return

    logger.info(f"待清理檔案數：{len(pending)}")
    for _, row in pending.iterrows():
        p = row["path"]
        try:
            date_key, item, file_name = _process_one_file(p)
        except Exception as e:
            # 你要求：遇錯立即中止（不繼續），把錯誤完整拋出
            logger.error(f"處理發生錯誤：{e}")
            raise

        # 更新 log（沿用舊設計：index=date, column=item, value=file_name）
        if isinstance(clean_log, pd.DataFrame):
            if date_key not in clean_log.index:
                clean_log.loc[date_key, item] = file_name
            else:
                clean_log.loc[date_key, item] = file_name
        else:
            # fallback：使用 set（極簡）
            done.add(file_name)

        # 即時存檔（避免中途中斷丟進度）
        pickleio(data=clean_log if isinstance(clean_log, pd.DataFrame) else done,
                 path=dbpath_clean_log, mode="save")
        logger.info(f"完成：{file_name}（date={date_key}, item={item}）")


# ---- CLI ----
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TWSE 資料清理器（合併版）")
    p.add_argument("--col", nargs="*", help="指定要清理的類別（資料夾名/collection key），預設全清")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    # logging 已預設 DEBUG
    process_twse_data(args.col)


if __name__ == "__main__":
    main()
