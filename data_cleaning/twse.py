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

import pandas as pd

# ---- StevenTricks 與 config ----
from StevenTricks.file_utils import pickleio
from StevenTricks.file_utils import PathWalk_df

from StevenTricks.convert_utils import safe_replace, safe_numeric_convert, stringtodate
from StevenTricks.dict_utils import keyinstr
from StevenTricks.internal_db import DBPkl

from config.conf import collection, fields_span, dropcol, key_set
from config.col_rename import colname_dic, transtonew_col
from config.col_format import numericol, datecol
from config.paths import dbpath_source, dbpath_cleaned, dbpath_cleaned_log  # 原始 pkl 的根目錄（crawler 產物 清洗後 SQLite 目錄 清洗完成 log（pickle）

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

    # 4) 日期欄位轉換（若有指定 datecol）
    date_cfg = datecol.get(item,{}).get(subitem,{})
    try:
        df = stringtodate(df,datecol=date_cfg, mode=3)  # 你專案原本使用 mode=3（常見 ROC/多格式）
    except Exception as e:
        # 報出第一個壞值、型別
        bad = df
        raise DataCleanError(
            "日期欄位轉換失敗",
            item=item, subitem=subitem, column=date_cfg,
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

# ---- 寫入資料庫 ----
def _db_path_for_item(item: str) -> str:
    _ensure_dir(dbpath_cleaned)
    return join(dbpath_cleaned, f"{item}")


def _write_to_db(df: pd.DataFrame, convert_mode="upcast", *, item: str, subitem: str) -> None:
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
    logger.debug(f"寫入 DB：{db_path} 表={subitem} PK={pk}")

    dbi = DBPkl(db_path, subitem)

    try:
        # 若你的 DBPkl.write_db 支援 convert_mode 參數就保留；否則移除
        dbi.write_db(df, convert_mode=convert_mode, primary_key=(pk if pk else None))
    except Exception as e:
        # === 把當前狀態丟到全域，方便 PyCharm 變數窗格檢視 ===
        global DEBUG_LAST_DF, DEBUG_LAST_CONTEXT
        DEBUG_LAST_DF = df  # 不做 copy，保留原物件，利於完整檢視
        conflict = getattr(dbi, "schema_conflict", None)  # 可能不存在

        # 盡量提供足夠的診斷資訊
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
            # 請小心 head/unique 可能很大；如資料量很大可移除這兩行
            "head": df.head(5),
            "dtypes": dtypes,
        }

        # 額外在 log 打印一下衝突（若有）
        if conflict:
            logger.debug(f"[DB schema conflict] {conflict}")

        # === 照你的策略：遇錯就停 ===
        raise



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
        _write_to_db(df1, item=parentdir, subitem=subitem)

    return date_key, parentdir, file_name


# ---- 清洗流程（可被 import 呼叫） ----
def process_twse_data(cols: Optional[List[str]] = None) -> None:
    """
    執行清洗流程。
    參數：
      cols: 要清的 collection 類別（資料夾名）；None 表示全部
    """
    _ensure_dir(dbpath_cleaned)
    # 載入或建立清洗 log（紀錄已處理檔名，避免重複清）
    if exists(dbpath_cleaned_log):
        clean_log = pickleio(path=dbpath_cleaned_log, mode="load")
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
        pickleio(data= clean_log if isinstance(clean_log, pd.DataFrame) else done,
                 path= dbpath_cleaned_log, mode="save")
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
    main([])

raw = {
    "fields": ["A","B"],
    "data": [[1,2]],
    "title": "主表",
    "groups": None,
    "tables": [
        {"fields1": ["X","Y"], "data1": [[9,8]], "subtitle": "子表一"},
        {"creditFields": ["C1","C2"], "creditList": [[3,4]], "creditTitle": "信用表"},
    ],
}
# 你的 key_set 如題
lst = key_extract(raw)
for i, d in enumerate(lst, 1):
    print(i, d.keys())  # 應能看到 fields/data/title/groups/notes 等鍵依規則被抽出




raw = pickleio(path=r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source/每日收盤行情/每日收盤行情_2023-09-25.pkl", mode="load")
raw1 = pickleio(path=r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/cleaned/每日收盤行情/每日收盤行情.pkl", mode="load")
raw2 = pickleio(path=r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/cleaned/每日收盤行情/每日收盤行情_schema.pkl", mode="load")