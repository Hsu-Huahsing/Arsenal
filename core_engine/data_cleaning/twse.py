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

    from data_cleaning.twse import process_twse_data

    # 清全部
    process_twse_data()

    # 或只清特定集合
    process_twse_data(["每日收盤行情"])

    import pandas as pd
    from data_cleaning.twse import clean_one_dataframe

    # 假資料：欄名會先經過 colname_dic 與 HTML 清理
    raw = pd.DataFrame(
        [["114/08/05", "2330", "10,000", "1,234.5"]],
        columns=["日期", "代號", "成交股數", "收盤價</br>"]
    )

    cleaned = clean_one_dataframe(
        raw,
        item="每日收盤行情",
        subitem="個股",
        date="2025-08-10"  # 若無日期欄，會用這個補 'date'
    )
    print(cleaned.dtypes)
    print(cleaned.head())


"""
import argparse
import logging
from os import makedirs
from os.path import exists, join, splitext, basename
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

# ---- StevenTricks 與 config ----
from StevenTricks.io.file_utils import pickleio
from StevenTricks.io.file_utils import PathWalk_df

from StevenTricks.core.convert_utils import safe_replace, safe_numeric_convert, stringtodate, keyinstr
from StevenTricks.db.internal_db import DBPkl

from config.conf import collection, fields_span, dropcol, key_set
from config.col_rename import colname_dic, transtonew_col
from config.col_format import numericol, datecol
from config.paths import (
    db_local_root,
)
# ---- 雲端 / 本機 路徑快取 ----
# 一開始載入時，config.paths 的 db_root 指向「雲端」，所以這三個就是雲端路徑
CLOUD_DBPATH_SOURCE = dbpath_source
CLOUD_DBPATH_CLEANED = dbpath_cleaned
CLOUD_DBPATH_CLEANED_LOG = dbpath_cleaned_log

# 本機根目錄（config.paths.path_dic["stock_twse_db"]["db_local"]）
LOCAL_DB_ROOT = db_local_root
LOCAL_DBPATH_SOURCE = LOCAL_DB_ROOT / "source"
LOCAL_DBPATH_CLEANED = LOCAL_DB_ROOT / "cleaned"
LOCAL_DBPATH_CLEANED_LOG = LOCAL_DBPATH_CLEANED / "log.pkl"

from StevenTricks.io.staging import staging_path

DEBUG_LAST_DF: Optional[pd.DataFrame] = None
DEBUG_LAST_CONTEXT: Dict[str, Any] = {}

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

# ---- job_state 設計 ----
# 統一 job_state 的欄位，不論新舊版本都走這一套
JOB_STATE_COLUMNS = [
    "file",              # 檔名（不含路徑）
    "path",              # 完整路徑
    "dir",               # 上層資料夾（TWSE 類別，例如 三大法人買賣超日報）
    "hash",              # 檔案 fingerprint（size + mtime）
    "source_mtime",      # 檔案最後異動時間
    "source_size",       # 檔案大小（bytes）
    "status",            # pending / success / failed
    "date",              # 清到的 date_key（例如 20251121）
    "item",              # item（例如 三大法人買賣超日報）
    "last_processed_at", # 我們成功/失敗寫入 DB 的時間
]

def _calc_file_state(path: str) -> Dict[str, Any]:
    """
    統一取得 source 檔案的目前狀態：
    - hash：用 size + mtime 組合而成，足夠判斷是否異動
    - source_size：檔案大小（bytes）
    - source_mtime：最後修改時間（pandas Timestamp）
    """
    p = Path(path)
    st = p.stat()
    size = st.st_size
    mtime = st.st_mtime  # float（秒）

    # fingerprint：size + 整數 mtime 字串
    fp = f"{size}-{int(mtime)}"

    return {
        "hash": fp,
        "source_size": size,
        "source_mtime": pd.Timestamp.fromtimestamp(mtime),
    }


def _get_span_cfg(item: str, subitem: str) -> Optional[dict]:
    """
    依序嘗試：
      1) fields_span[item][subitem]
      2) fields_span[subitem]
    有哪個就用哪個；都沒有回傳 None。
    """
    by_item = (fields_span.get(item, {}) or {}).get(subitem)
    direct  = fields_span.get(subitem)
    return by_item or direct

def _is_partition_by_date_item(item: str) -> bool:
    """
    判斷此 item 是否為「日頻率」的日報表：
    - 依據 config.conf.collection[item]['freq']
    - 只要 freq 是 'D' / 'd' 或類似 '1D'，就視為按 date 做 partition 覆寫
    """
    cfg = collection.get(item) or {}
    freq = cfg.get("freq")
    if freq is None:
        return False

    # 統一成字串判斷，避免大小寫問題或 '1D' 之類寫法
    s = str(freq).strip().upper()
    if s == "D":
        return True
    # 如果你未來想支援 '1D'、'DAY' 之類，也可以順便打開：
    if s in {"1D", "DAY", "DAILY"}:
        return True

    return False

def _make_bucket_key(date_series: pd.Series, mode: str) -> pd.Series:
    """
    把 date 欄位轉成「bucket key」，依 mode 回傳字串 Series：
      - all     → 全部同一 bucket（不應進來，呼叫端會先略過）
      - year    → '2020'
      - quarter → '2020Q1'
      - month   → '2020-01'
      - day     → '2020-01-31'
    """
    mode = (mode or "all").lower()

    if not pd.api.types.is_datetime64_any_dtype(date_series):
        # 防呆：如果不是 datetime，就硬轉一次
        date_series = pd.to_datetime(date_series)

    if mode == "year":
        return date_series.dt.strftime("%Y")

    if mode == "quarter":
        # to_period('Q') 會產生類似 '2020Q1'
        return date_series.dt.to_period("Q").astype(str)

    if mode == "month":
        return date_series.dt.strftime("%Y-%m")

    if mode == "day":
        return date_series.dt.strftime("%Y-%m-%d")

    # 其他（含 all）呼叫端不應進來；這裡直接給同一個 key
    return pd.Series(["ALL"] * len(date_series), index=date_series.index)

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


def _list_source_pickles(root: str) -> pd.DataFrame:
    """
    列出 root 下所有 .pkl 檔；回傳 DataFrame 包含 columns: file, path, dir（上層資料夾名）
    優先用 StevenTricks.PathWalk_df
    """
    df = PathWalk_df(root, [], ["log"], [".DS_Store","productlist"], [".pkl"])  # 依你的慣例
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


def _load_job_state() -> pd.DataFrame:
    """
    從 dbpath_cleaned_log 載入 job_state：
    - 若不存在 → 回傳空 DataFrame（含固定欄位）
    - 若是舊版 log（set/list/DataFrame）→ 自動補齊欄位
    """
    if not exists(dbpath_cleaned_log):
        return pd.DataFrame(columns=JOB_STATE_COLUMNS)

    obj = pickleio(path=dbpath_cleaned_log, mode="load")

    if isinstance(obj, pd.DataFrame):
        js = obj.copy()
        # 補齊缺少欄位
        for col in JOB_STATE_COLUMNS:
            if col not in js.columns:
                js[col] = pd.NA
        return js[JOB_STATE_COLUMNS]

    # 舊版：set/list 只記 file 名稱
    if isinstance(obj, (set, list, tuple)):
        return pd.DataFrame(
            {"file": list(map(str, obj))},
            columns=JOB_STATE_COLUMNS,
        )

    # 其他未知格式：當作空
    return pd.DataFrame(columns=JOB_STATE_COLUMNS)


def _save_job_state(job_state: pd.DataFrame) -> None:
    """
    將 job_state 存回 dbpath_cleaned_log。
    確保欄位順序與 JOB_STATE_COLUMNS 一致。
    """
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
    """
    新增或更新一筆 job_state 紀錄（以 path 當唯一鍵）。

    參數：
      - file：檔名（不含路徑）
      - path：完整路徑
      - dir_name：上層資料夾名（TWSE 類別）
      - date_key：本次清理得到的 date（若還沒拿到可給 None）
      - item：TWSE item 名稱（例如 三大法人買賣超日報，若還沒拿到可給 None）
      - state：要覆寫的欄位 dict，例如：
          {"status": "pending", "hash": "...", "source_mtime": ts, ...}
    """
    if job_state.empty:
        idx = pd.Series([], dtype=bool)
    else:
        idx = (job_state["path"] == path)

    if not idx.any():
        # 新紀錄
        row = {col: pd.NA for col in JOB_STATE_COLUMNS}
        row.update(
            {
                "file": file,
                "path": path,
                "dir": dir_name,
                "date": date_key,
                "item": item,
            }
        )
        row.update(state)

        # 👇 修正這裡，避免「空 DataFrame + concat」造成 FutureWarning
        row_df = pd.DataFrame([row], columns=JOB_STATE_COLUMNS)

        if job_state.empty:
            # 第一次直接用 row_df 當起始 job_state
            job_state = row_df
        else:
            # 後續才用 concat 疊上去
            job_state = pd.concat([job_state, row_df], ignore_index=True)

    else:
        # 更新既有紀錄（原來這段保持不動）
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



# ---- 解析 TWSE API 結構 → 子表 dict list ----
def key_extract(dic: dict) -> list[dict]:
    """
    依據全域 key_set 從 raw dict 擷取多個「子表」片段 (fields/data/title/...)，
    同時支援：
      - step == "main1"：帶序號切片（如 fields, fields1, fields2, ...）
      - 其他 step（如 "set1"）：一次聚合
      - raw["tables"]：若存在且為 list，逐一用相同規則抽取
    回傳：list[dict]，例如 [{"fields":..., "data":..., "title":..., "groups":..., "notes":...}, ...]
    """
    if not isinstance(dic, dict):
        raise TypeError(f"key_extract() expects dict, got {type(dic).__name__}")

    out: list[dict] = []

    def _listify(x):
        # 設定允許寫成字串或清單；統一轉清單
        return x if isinstance(x, (list, tuple)) else [x]

    def _find_first_key(container: dict, aliases: list[str]) -> tuple[str, bool]:
        """
        在 container 裡依序找第一個存在的別名鍵；回傳 (命中的鍵名, 是否命中)
        """
        for k in aliases:
            if k in container:
                return k, True
        return "", False

    def _extract_from_container(container: dict) -> list[dict]:
        """
        依 key_set 規則，從單一 container (通常是 raw 或 raw 的一個 table dict) 抽出子表。
        """
        dicts: list[dict] = []
        for step, set_i in key_set.items():
            if not isinstance(set_i, dict):
                continue

            if step == "main1":
                # 完全復刻你原本的停止條件：cnt > 1 且沒命中就停止
                cnt = 0
                while True:
                    curr: dict = {}
                    for key_name, alias_list in set_i.items():
                        aliases = _listify(alias_list)
                        # cnt==0 用原名；cnt>0 用 f"{alias}{cnt}"
                        candidates = [a if cnt == 0 else f"{a}{cnt}" for a in aliases]
                        hit_key, ok = _find_first_key(container, candidates)
                        if ok:
                            curr[key_name] = container[hit_key]

                    if curr:
                        dicts.append(curr)
                    elif cnt > 1:
                        # 與原碼邏輯等價：cnt>1 且無命中 → break
                        break

                    cnt += 1

            else:
                # 非 main1：只做一次聚合
                curr: dict = {}
                for key_name, alias_list in set_i.items():
                    aliases = _listify(alias_list)
                    hit_key, ok = _find_first_key(container, aliases)
                    if ok:
                        curr[key_name] = container[hit_key]
                if curr:
                    dicts.append(curr)

        return dicts

    # 1) 先從 raw 本體抽
    out.extend(_extract_from_container(dic))

    # 2) 若 raw 中還有 tables（多表），逐一處理
    tables = dic.get("tables")
    if isinstance(tables, list):
        for t in tables:
            if isinstance(t, dict):
                out.extend(_extract_from_container(t))

    return out


# ---- 兩種 DataFrame 組裝 ----
def frameup_safe(d: Dict[str, Any]) -> pd.DataFrame:
    """
    無群組欄位：直接以 d['fields'] 對齊 d['data']。
    若 data 的欄數 > fields，超出者丟棄（結構噪音），但不丟列。
    """
    fields = list(d.get("fields",[]))
    rows = list(d.get("data",[]))
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
    fields = list(d.get("fields",[]))
    rows = list(d.get("data",[]))
    if not fields or not rows:
        raise DataCleanError("data_cleaned_groups：缺少 fields 或 data")
    groups = span_cfg.get("groups")
    if not groups:
        raise DataCleanError("data_cleaned_groups：span_cfg 缺少 groups")
    # 計算每群組欄位數，產生目標欄名
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
    df = df.drop(columns=dropcol, errors="ignore")
    # 2) 欄名舊→新（細項規則）
    rename_cfg = transtonew_col.get(item,{}).get(subitem,{})
    if rename_cfg:
        df = df.rename(columns=rename_cfg)
    # 3) 數值欄位轉換
    num_cfg = numericol.get(item,{}).get(subitem,{})
    df = safe_numeric_convert(df, num_cfg)

    # 4) 若沒有任何日期欄，補一個統一 'date'
    if "date" not in df.columns:
        df.insert(0, "date", date_key)

    # 5) 日期欄位轉換（若有指定 datecol）
    date_cfg = datecol.get(item,{}).get(subitem,["date"])
    try:
        df = stringtodate(df, datecol=date_cfg, mode=4)
    except Exception as e:
        # 報出第一個壞值、型別
        bad = df
        raise DataCleanError(
            "日期欄位轉換失敗",
            item=item, subitem=subitem, column=date_cfg,
            value=bad, value_type=type(bad).__name__,
            hint="請補充 changetype_stringtodate 規則或前置清理邏輯",
        ) from e


    # 6) 欄位順序微整：把常見鍵放前面
    front = [c for c in ["date", "代號","名稱"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    return df

# ---- 寫入資料庫 ----
def _db_path_for_item(item: str) -> str:
    _ensure_dir(dbpath_cleaned)
    return join(dbpath_cleaned, f"{item}")


def _write_to_db(
    df: pd.DataFrame,
    convert_mode: str = "upcast",
    *,
    item: str,
    subitem: str,
    bucket_mode: str = "all",     # ★ 新增
) -> None:
    """
    預設 PK 規則：
      - 同時有 '代號' 與 'date' → ['代號','date']
      - 僅有 'date' → ['date']
      - 否則不設 PK（交由 DBPkl 處理）
    """
    pk: List[str] = []
    if "代號" in df.columns and "date" in df.columns:
        pk = ["代號", "date"]
    elif "名稱" in df.columns and "date" in df.columns:
        pk = ["名稱", "date"]
    elif "date" in df.columns:
        pk = ["date"]

    db_path = _db_path_for_item(item)

    partition_by_date = "date" in df.columns and _is_partition_by_date_item(item)

    logger.debug(
        "寫入 DB：%s 表=%s PK=%s partition_by_date=%s bucket_mode=%s",
        db_path,
        subitem,
        pk,
        partition_by_date,
        bucket_mode,
    )

    # ---- ★ 若是日頻率 + bucket_mode != all → 依 bucket 拆表 ----
    if partition_by_date and bucket_mode.lower() != "all":
        # 產生 bucket key（字串）
        bucket_key = _make_bucket_key(df["date"], bucket_mode)

        # 一個 bucket 對應一個實際 table，例如：
        #   subitem='三大法人買賣超日報', bucket='2020-01'
        #   → table_name='三大法人買賣超日報__2020-01'
        for b, df_chunk in df.groupby(bucket_key):
            table_name = f"{subitem}__{b}"

            logger.debug(
                "寫入分桶表：%s/%s（bucket=%s, rows=%d）",
                db_path,
                table_name,
                b,
                len(df_chunk),
            )

            # ★ data 檔名用 table_name（會帶 __2012 等），
            #   schema 一律用 logical_table_name=subitem
            dbi = DBPkl(
                db_path,
                table_name,
                logical_table_name=subitem,
            )


            try:
                dbi.write_partition(
                    df_chunk,
                    convert_mode=convert_mode,
                    partition_cols=["date"],
                    primary_key=(pk if pk else None),
                )
            except Exception as e:
                # debug 區塊照舊，但注意用 table_name
                global DEBUG_LAST_DF, DEBUG_LAST_CONTEXT
                DEBUG_LAST_DF = df_chunk
                conflict = getattr(dbi, "schema_conflict", None)
                try:
                    dtypes = df_chunk.dtypes.astype(str).to_dict()
                except Exception:
                    dtypes = {}

                DEBUG_LAST_CONTEXT = {
                    "item": item,
                    "subitem": table_name,
                    "db_path": str(db_path),
                    "pk": pk,
                    "convert_mode": convert_mode,
                    "conflict": conflict,
                    "exception_type": type(e).__name__,
                    "exception_str": str(e),
                    "columns": list(df_chunk.columns),
                    "shape": tuple(df_chunk.shape),
                    "head": df_chunk.head(5),
                    "dtypes": dtypes,
                }
                if conflict:
                    logger.debug(f"[DB schema conflict] {conflict}")
                raise

        # 分桶模式下，這個函式到這裡就結束，不再走下面的「單一表」邏輯
        return


    # ---- ★ 否則維持原本單一表行為 ----
    dbi = DBPkl(db_path, subitem, logical_table_name=subitem)


    try:
        if partition_by_date:
            dbi.write_partition(
                df,
                convert_mode=convert_mode,
                partition_cols=["date"],
                primary_key=(pk if pk else None),
            )
        else:
            dbi.write_db(
                df,
                convert_mode=convert_mode,
                primary_key=(pk if pk else None),
            )

    except Exception as e:
        # 原本 debug 區塊原封不動

        DEBUG_LAST_DF = df
        conflict = getattr(dbi, "schema_conflict", None)
        try:
            dtypes = df.dtypes.astype(str).to_dict()
        except Exception:
            dtypes = {}

        DEBUG_LAST_CONTEXT = {
            "item": item,
            "subitem": subitem,
            "db_path": str(db_path),
            "pk": pk,
            "convert_mode": convert_mode,
            "conflict": conflict,
            "exception_type": type(e).__name__,
            "exception_str": str(e),
            "columns": list(df.columns),
            "shape": tuple(df.shape),
            "head": df.head(5),
            "dtypes": dtypes,
        }
        if conflict:
            logger.debug(f"[DB schema conflict] {conflict}")
        raise
        # === 照你的策略：遇錯就停 ===



# ---- 清洗一個檔案（主流程子步驟） ----
def _process_one_file(
    file_path: str,
    *,
    bucket_mode: str = "all",   # ★ 新增
) -> Tuple[str, str, str]:
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

    base, _ = splitext(file_name)

    # 取 crawler 取得日
    try:
        date_key = raw.get("crawlerdic",{}).get("payload",{}).get("date")
    except Exception as e:
        raise DataCleanError("無法取得 crawler 日期", file=file_name, item=parentdir, value=raw.get("crawlerdic")) from e

    # 決定允許的子表（標準化後）
    # 以 crawler 的 subtitle 優先，否則取 config.collection[item]['subtitle']
    subtitle_from_crawler = raw.get("crawlerdic",{}).get("subtitle")
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
    # 取所有子表（title, fields, data）
    sub_tables = key_extract(raw)

    # 如果完全抓不到子表，先判斷是「沒資料日」還是「格式異常」
    if not sub_tables:
        stat_msg = raw.get("stat") or raw.get("note") or ""

        # 典型情況：TWSE 回傳「很抱歉，沒有符合條件的資料!」或類似字眼
        if isinstance(stat_msg, str) and (
            "沒有符合條件的資料" in stat_msg or
            "查無資料" in stat_msg
        ):
            logger.warning(
                f"略過檔案（該日無資料）：file={file_name}, item={parentdir}, stat={stat_msg!r}"
            )
            # 這邊直接當作「空資料日」，讓流程繼續跑其他檔案
            return date_key, parentdir, file_name

        # 其他情況 → 真的找不到資料表，維持原本嚴格錯誤策略
        raise DataCleanError(
            "未找到任何可清理的子表",
            file=file_name,
            item=parentdir,
            value=list(raw.keys()),  # 多給你 raw 的 key 幫助之後 debug
        )

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
            span_cfg = _get_span_cfg(parentdir, subitem)

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
        _write_to_db(
            df1,
            item=parentdir,
            subitem=subitem,
            bucket_mode=bucket_mode,
        )
    return date_key, parentdir, file_name


# ---- 清洗流程（可被 import 呼叫） ----

def _process_twse_data_impl(
            cols: Optional[List[str]] = None,
            max_files_per_run: Optional[int] = None,# 每輪最多處理幾個「實際清理」的檔案
            bucket_mode: str = "all",  # ★ 新增
    ) -> int:
    """
    真正執行清理邏輯的內部函式。

    回傳值：
        本輪「實際有執行 _process_one_file」的檔案數（略過的不算）。

    注意：這裡假設 dbpath_cleaned / dbpath_cleaned_log 已經是「要寫入的那個路徑」
          （可能是 iCloud，可能是本機 staging，由外層負責決定）。
    """
    _ensure_dir(dbpath_cleaned)

    # 1) 載入 job_state
    job_state = _load_job_state()

    # 2) 列出所有 source pkl 檔
    files_df = _list_source_pickles(dbpath_source)

    # 若有指定要清的類別，先過濾
    if cols:
        files_df = files_df[files_df["dir"].isin(cols)].copy()

    total_files = len(files_df)
    if files_df.empty:
        logger.info("找不到任何待處理的 source 檔案。")
        return 0

    logger.info(f"待檢查檔案數：{total_files}")

    # === 進度統計用 ===
    processed = 0         # 本輪實際有清理幾檔
    start_time = datetime.now()

    # 3) 逐檔決定：略過 / 重跑
    for scanned_idx, (_, row) in enumerate(files_df.iterrows(), start=1):

        # 若有設定 max_files_per_run，且已達上限 → 提前結束本輪
        if max_files_per_run is not None and processed >= max_files_per_run:
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
        dir_name  = row["dir"]

        # 3-1) 取得 source 當前狀態（size / mtime / hash）
        state_now = _calc_file_state(file_path)
        fp_now    = state_now["hash"]
        mtime_now = state_now["source_mtime"]
        size_now  = state_now["source_size"]

        # 3-2) 找出 job_state 既有紀錄
        rec_idx = (job_state["path"] == file_path) if not job_state.empty else pd.Series([], dtype=bool)
        rec = job_state.loc[rec_idx].iloc[0] if rec_idx.any() else None

        # 3-3) 判斷是否可以安全略過
        if rec is not None:
            rec_status = rec.get("status")
            rec_hash   = rec.get("hash")
            rec_mtime  = rec.get("source_mtime")

            # status = success 且 hash/mtime 完全一致 → 當作沒變化，略過
            if (
                rec_status == "success"
                and pd.notna(rec_hash)
                and rec_hash == fp_now
                and pd.notna(rec_mtime)
                and pd.Timestamp(rec_mtime) == mtime_now
            ):
                logger.debug(f"略過檔案（source 未變更）：{file_name}")
                continue

            # status 是 success 但 hash/mtime 改變 → 防呆：改成 pending
            if rec_status == "success" and (
                rec_hash != fp_now
                or (pd.notna(rec_mtime) and pd.Timestamp(rec_mtime) != mtime_now)
            ):
                logger.warning(
                    "偵測到 source 在上次成功清理後有變更，標記為 pending：file=%s, old_mtime=%s, new_mtime=%s",
                    file_name,
                    rec_mtime,
                    mtime_now,
                )
                job_state.loc[rec_idx, "status"] = "pending"

        # 3-4) 進入清理前，先標記 pending
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

        # 3-5) 實際執行清理
        try:
            date_key, item, cleaned_file_name = _process_one_file(
                file_path,
                bucket_mode=bucket_mode,
            )
        except Exception as e:
            # 標記為 failed
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
            logger.error(f"處理發生錯誤：{e}")
            # 按你的策略：直接整體中止
            raise

        # 3-6) 清理成功 → 更新 job_state
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

        logger.info(
            "完成：%s（date=%s, item=%s）｜本輪已處理 %d 檔 / 總檔數 %d（掃描到第 %d 檔），平均耗時 %.1f 秒/檔。",
            cleaned_file_name,
            date_key,
            item,
            processed,
            total_files,
            scanned_idx,
            avg_sec,
        )

    return processed

def process_twse_data(
    cols: Optional[List[str]] = None,
    *,
    storage_mode: str = "cloud",     # "cloud" / "cloud_staging" / "local"
    batch_size: Optional[int] = None,
    bucket_mode: str = "all",
) -> None:
    """
    TWSE 清理主入口。

    cols:
        要清哪幾個 item（如 ["三大法人買賣超日報"]），None 則清全部。

    storage_mode:
        - "cloud"         : 直接用雲端 db_root（config.paths 的 db_root）
        - "cloud_staging" : 雲端 + 本機暫存（先把 cleaned 整包拉到本機處理，再同步回雲端）
        - "local"         : 完全只用本機 root（config.paths 的 db_local_root），不碰雲端

    batch_size:
        - 僅在 "cloud_staging" 模式有效，每一輪最多處理幾個檔案。
          例如 500 表示每次 staging 只處理 500 個 source 檔，處理完同步回雲端，再下載下一批。
          None 則視為「一次清到底」。
    """
    global dbpath_source, dbpath_cleaned, dbpath_cleaned_log

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

    # 先備份「目前」的 active 路徑（通常是雲端）
    orig_source = dbpath_source
    orig_cleaned = dbpath_cleaned
    orig_cleaned_log = dbpath_cleaned_log

    # ---------- 情境 A：完全本機模式 ----------
    if storage_mode == "local":
        # 切換成「本機」路徑
        dbpath_source = LOCAL_DBPATH_SOURCE
        dbpath_cleaned = LOCAL_DBPATH_CLEANED
        dbpath_cleaned_log = LOCAL_DBPATH_CLEANED_LOG

        _ensure_dir(str(dbpath_source))
        _ensure_dir(str(dbpath_cleaned))

        logger.info(
            "進入 LOCAL 模式：source=%s, cleaned=%s",
            dbpath_source,
            dbpath_cleaned,
        )

        try:
            _process_twse_data_impl(
                cols,
                bucket_mode=bucket_mode,
                max_files_per_run=batch_size,
            )
        finally:
            # 不管有沒有出錯，都把路徑還原
            dbpath_source = orig_source
            dbpath_cleaned = orig_cleaned
            dbpath_cleaned_log = orig_cleaned_log

        return

    # ---------- 情境 B / C：以雲端為主 ----------
    # 先把 active 路徑切回「雲端版本」
    dbpath_source = CLOUD_DBPATH_SOURCE
    dbpath_cleaned = CLOUD_DBPATH_CLEANED
    dbpath_cleaned_log = CLOUD_DBPATH_CLEANED_LOG

    # B-1：純雲端（舊的「不用 staging」）
    if storage_mode == "cloud":
        logger.info(
            "進入 CLOUD (no staging) 模式：source=%s, cleaned=%s",
            dbpath_source,
            dbpath_cleaned,
        )
        try:
            _process_twse_data_impl(
                cols,
                bucket_mode=bucket_mode,
                max_files_per_run=batch_size,
            )
        finally:
            dbpath_source = orig_source
            dbpath_cleaned = orig_cleaned
            dbpath_cleaned_log = orig_cleaned_log
        return

    # B-2：雲端 + 本機 staging（你原本的 use_local_db_staging=True 模式）
    # 等同以前的程式，但邏輯搬到這裡而且更明確
    if batch_size is None:
        batch_size = 10_000_000  # 一次清到底

    target_cleaned: Path = CLOUD_DBPATH_CLEANED
    staging_root: Path = db_local_root

    batch_no = 0
    while True:
        batch_no += 1
        logger.info("===== 開始 staging batch %d，batch_size=%d =====", batch_no, batch_size)

        # staging_path 會：
        # 1) 把「雲端 cleaned」整個複製到 staging_root 下某個 staging_xxx/cleaned 資料夾
        # 2) yield 本機 cleaned 的路徑
        # 3) 離開 with 時把本機結果同步回雲端，並把 staging_xxx 刪掉
        with staging_path(target_cleaned, enable=True, staging_root=staging_root) as local_cleaned:
            try:
                dbpath_cleaned = local_cleaned
                dbpath_cleaned_log = local_cleaned / "log.pkl"

                logger.info(
                    "cloud_staging 模式：本輪在本機 cleaned=%s 上處理",
                    dbpath_cleaned,
                )

                processed = _process_twse_data_impl(
                    cols,
                    bucket_mode=bucket_mode,
                    max_files_per_run=batch_size,
                )
            finally:
                # 恢復成雲端 cleaned
                dbpath_cleaned = CLOUD_DBPATH_CLEANED
                dbpath_cleaned_log = CLOUD_DBPATH_CLEANED_LOG

        if processed == 0:
            logger.info("沒有待處理檔案，staging 迴圈結束。")
            break

        logger.info("===== staging batch %d 完成，本輪處理 %d 個檔案 =====", batch_no, processed)

    # 最後保險再把 active 路徑恢復到原本狀態
    dbpath_source = orig_source
    dbpath_cleaned = orig_cleaned
    dbpath_cleaned_log = orig_cleaned_log


# ---- CLI ----
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TWSE 資料清理器（合併版）")
    p.add_argument(
        "--col",
        nargs="*",
        help="指定要清理的類別（資料夾名/collection key），預設全清",
    )
    p.add_argument(
        "--storage-mode",
        type=str,
        default="cloud",
        choices=["cloud", "cloud_staging", "local"],
        help="資料儲存模式：cloud=直接用雲端；cloud_staging=雲端+本機暫存；local=完全只用本機 db_local_root。",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每一輪 staging 要處理的最大檔案數（例如 500）",
    )
    p.add_argument(
        "--bucket-mode",
        choices=["all", "year", "quarter", "month", "day"],
        default="all",
        help="日期分桶模式：all=整檔一個表、year=每年一表、quarter=每季一表、month=每月一表、day=每日一表",
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
    )


if __name__ == "__main__":
    # 讓 _parse_args 自己去處理 sys.argv（含 PyCharm 的垃圾參數）
    main()
#
# raw = {
#     "fields": ["A","B"],
#     "data": [[1,2]],
#     "title": "主表",
#     "groups": None,
#     "tables": [
#         {"fields1": ["X","Y"], "data1": [[9,8]], "subtitle": "子表一"},
#         {"creditFields": ["C1","C2"], "creditList": [[3,4]], "creditTitle": "信用表"},
#     ],
# }
# # 你的 key_set 如題
# lst = key_extract(raw)
# for i, d in enumerate(lst, 1):
#     print(i, d.keys())  # 應能看到 fields/data/title/groups/notes 等鍵依規則被抽出
#
#
#
#
# raw = pickleio(path=r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source/三大法人買賣超日報/三大法人買賣超日報_2023-09-25.pkl", mode="load")
# raw1 = pickleio(path=r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/cleaned/三大法人買賣超日報/三大法人買賣超日報.pkl", mode="load")
# raw2 = pickleio(path=r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/cleaned/三大法人買賣超日報/三大法人買賣超日報_schema.pkl", mode="load")