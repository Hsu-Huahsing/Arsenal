# -*- coding: utf-8 -*-
""" Schema Utilities
處理資料結構分析與欄位重建，如多層 dict 拆解、欄位命名分類、表格重建等。
依賴 StevenTricks 中的欄位定義模組（colname_dic, numericol 等）。
"""

import logging
import re
from typing import Tuple
from os.path import exists, dirname
from os import makedirs
from shutil import copy2
from config.paths import dbpath_log
from StevenTricks.io.file_utils import pickleio
from pathlib import Path
from typing import List, Optional, Iterable, Dict, Any
import datetime as _dt
import pandas as pd

# 類別對應鍵，用來分類資料欄位的 key
productkey = {
    "col": ["field"],
    "value": ["data", "list"],
    "title": ["title"]
}

def getkeys(data):
    """ 從 JSON 字典資料中歸類每一層的 key 到 col/value/title 三大類 """
    productcol = {
        "col": [],
        "value": [],
        "title": [],
    }
    for key in sorted(data.keys()):
        for k, i in productkey.items():
            i = [key for _ in i if _ in key.lower()]
            if i:
                productcol[k] += i
    return pd.DataFrame(productcol)


def productdict(source, keydf):
    """ 根據欄位分類表（由 getkeys() 回傳）產生對應的子表格 """
    productdict = {}
    for col, value, title in keydf.values:
        if not source[value]:
            continue
        df = pd.DataFrame(data=source[value], columns=source[col])
        productdict[source[title]] = df
    return productdict


def warehouseinit(path):
    """Initialize warehouse folder with 'source' and 'cleaned' subfolders."""
    for sub in ['source', 'cleaned']:
        makedirs(Path(path) / sub, exist_ok=True)


def safe_frameup_data(data_dict={}, fields=[]):
    data = pd.DataFrame(data_dict)
    col_other = list(range(0,len(data.columns)-len(fields),1))
    data.columns = fields + col_other
    data = data.drop(col_other, axis=1)
    return data

# === log.pkl 維護工具（精簡版 class；貼到 schema_utils.py 末尾） ===

_log = logging.getLogger(__name__)
_YMD = re.compile(r"^\d{4}-\d{2}-\d{2}$")

class LogMaintainer:
    """log.pkl 維護工具：讀取 / 檢視 / 驗證 / 正規化 / 刪除 / 改名。"""

    def __init__(self, path: str = str(dbpath_log), *, backup: bool = True, logger: Optional[logging.Logger] = None):
        self.path = path
        self.backup = backup
        self.logger = logger or _log
        self._df: Optional[pd.DataFrame] = None  # lazy-loaded cache

    # ---------- 內部工具 ----------
    def _timestamp(self) -> str:
        return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _backup(self) -> Optional[str]:
        if not exists(self.path):
            return None
        dst = f"{self.path}.bak.{self._timestamp()}"
        makedirs(dirname(dst), exist_ok=True)
        copy2(self.path, dst)
        self.logger.info(f"[log] 已備份：{dst}")
        return dst

    @staticmethod
    def _nanlike(x: Any) -> bool:
        return (not isinstance(x, str) and pd.isna(x)) or (isinstance(x, str) and x.strip().lower() in {"", "nan", "nat"})

    @staticmethod
    def _to_key(label: Any) -> Optional[str]:
        """可解析日期 → 'YYYY-MM-DD'；否則回傳 None。"""
        ts = pd.to_datetime([label], errors="coerce")
        if pd.isna(ts[0]):
            return None
        return ts[0].strftime("%Y-%m-%d")

    @staticmethod
    def _is_valid_key(label: Any) -> bool:
        """是否為字串且匹配 YYYY-MM-DD，且為有效日。"""
        if not isinstance(label, str) or not _YMD.match(label):
            return False
        try:
            pd.to_datetime(label, format="%Y-%m-%d", errors="raise")
            return True
        except Exception:
            return False

    # ---------- I/O ----------
    def load(self, *, force: bool = False) -> pd.DataFrame:
        if self._df is not None and not force:
            return self._df
        if not exists(self.path):
            raise FileNotFoundError(f"log.pkl 不存在：{self.path}")
        df = pickleio(path=self.path, mode="load")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"log.pkl 內容不是 DataFrame：{type(df)}")
        self._df = df
        return self._df

    def df(self) -> pd.DataFrame:
        return self.load()

    def save(self) -> None:
        if self._df is None:
            raise RuntimeError("沒有可儲存的資料（請先呼叫 load() 或指派 self._df）")
        if self.backup:
            self._backup()
        makedirs(dirname(self.path), exist_ok=True)
        pickleio(data=self._df, path=self.path, mode="save")
        self.logger.info(f"[log] 已寫入：{self.path}")

    # ---------- 驗證 / 正規化 ----------
    def validate_index(self) -> Dict[str, Any]:
        df = self.df()
        idx_ts = pd.to_datetime(df.index, errors="coerce")
        bad_mask = idx_ts.isna()
        dup_mask = pd.Index(df.index).duplicated(keep=False)
        return {
            "total": len(df),
            "bad_count": int(bad_mask.sum()),
            "bad_labels": list(df.index[bad_mask]),
            "duplicate_count": int(dup_mask.sum()),
            "duplicate_labels": list(pd.Index(df.index)[dup_mask].unique()),
        }

    def normalize_index(self, *, in_place: bool = True) -> Tuple[pd.DataFrame, List[Tuple[Any, str]]]:
        base = self.df().copy()
        idx_ts = pd.to_datetime(base.index, errors="coerce")
        new_labels, changed = [], []
        for orig, ts in zip(base.index, idx_ts):
            if pd.isna(ts):
                new_labels.append(orig)
            else:
                key = ts.strftime("%Y-%m-%d")
                new_labels.append(key)
                if key != orig:
                    changed.append((orig, key))
        base.index = pd.Index(new_labels)
        if in_place:
            self._df = base
        return base, changed

    # ---------- 查詢 ----------
    def show_rows(self, labels: Iterable[Any]) -> pd.DataFrame:
        df = self.df()
        out = []
        for lab in labels:
            # 先原樣取
            try:
                out.append(df.loc[[lab]])
                continue
            except Exception:
                pass
            # 再用標準 key 取
            key = self._to_key(lab)
            if key and key in df.index:
                out.append(df.loc[[key]])
        return pd.concat(out) if out else pd.DataFrame()

    # ---------- 列出不合規索引 ----------
    def list_invalid_index(self) -> pd.DataFrame:
        """
        列出所有「不是文字 yyyy-mm-dd（且為有效日期）」的索引。
        欄位：label, label_repr, type, is_str, matches_ymd, valid_date, wait_cols
        """
        df = self.df()
        # 向量化初篩
        idx = pd.Index(df.index)
        is_str = idx.map(lambda x: isinstance(x, str))
        matches = idx.map(lambda x: bool(_YMD.match(x)) if isinstance(x, str) else False)
        # 只對 matches 的做嚴格日期驗證
        valid_date = []
        for x, m in zip(idx, matches):
            if m:
                try:
                    pd.to_datetime(x, format="%Y-%m-%d", errors="raise")
                    valid_date.append(True)
                except Exception:
                    valid_date.append(False)
            else:
                valid_date.append(False)

        report = pd.DataFrame({
            "label": idx,
            "label_repr": idx.map(repr),
            "type": idx.map(lambda x: type(x).__name__),
            "is_str": is_str,
            "matches_ymd": matches,
            "valid_date": valid_date,
        })

        invalid = report[~(report["is_str"] & report["matches_ymd"] & report["valid_date"])].copy()

        # 補 wait_cols（逐列查一次）
        wait_list = []
        for lab in invalid["label"]:
            row = None
            try:
                row = df.loc[[lab]]
            except Exception:
                key = self._to_key(lab)
                if key and key in df.index:
                    row = df.loc[[key]]
            if row is None or row.empty:
                wait_list.append("")
            else:
                wait_cols = [c for c, v in row.iloc[0].items() if str(v) == "wait"]
                wait_list.append(",".join(wait_cols))
        invalid["wait_cols"] = wait_list

        invalid.sort_values(by=["is_str", "matches_ymd", "valid_date", "label_repr"],
                            ascending=[True, True, True, True], inplace=True)
        return invalid

    # ---------- 刪除 ----------
    def delete_labels(self, labels: Iterable[Any], *, in_place: bool = True) -> pd.DataFrame:
        """
        刪除指定索引（支援原樣、可解析、NaN 類）；labels 由你人工挑選。
        """
        df = self.df()
        before = len(df)
        to_drop = set()

        # 原樣命中
        for lab in labels:
            if lab in df.index:
                to_drop.add(lab)

        # 可解析命中
        for lab in labels:
            key = self._to_key(lab)
            if key and key in df.index:
                to_drop.add(key)

        # NaN 類 → 一次清掉所有不可解析索引
        if any(self._nanlike(lab) for lab in labels):
            idx_ts = pd.to_datetime(df.index, errors="coerce")
            to_drop.update(df.index[idx_ts.isna()])

        new_df = df.drop(index=list(to_drop), errors="ignore")
        self.logger.info(f"[log] 刪除 {len(to_drop)} 列；{before} → {len(new_df)}")
        if in_place:
            self._df = new_df
        return new_df

    def delete_bad_date_indices(self, *, in_place: bool = True) -> pd.DataFrame:
        """刪除所有無法解析為日期的索引。"""
        df = self.df()
        bad_mask = pd.to_datetime(df.index, errors="coerce").isna()
        if not bad_mask.any():
            self.logger.info("[log] 沒有發現壞的日期索引。")
            return df.copy()
        bad = list(df.index[bad_mask])
        self.logger.warning(f"[log] 將刪除 {len(bad)} 筆無效日期索引（示例：{bad[:5]} ...）")
        new_df = df.drop(index=bad, errors="ignore")
        if in_place:
            self._df = new_df
        return new_df

    # ---------- 改名 ----------
    def rename_label(self, old_label: Any, new_date_str: str, *, overwrite: bool = False, in_place: bool = True) -> pd.DataFrame:
        """將舊索引改名為合法 'YYYY-MM-DD'。支援 old_label 為字串/Timestamp/NaN。"""
        if not self._is_valid_key(new_date_str):
            raise ValueError(f"新索引不是合法 'YYYY-MM-DD'：{new_date_str}")
        df = self.df()

        # 找出實際存在的舊索引
        candidate = None
        if old_label in df.index:
            candidate = old_label
        else:
            key = self._to_key(old_label)
            if key and key in df.index:
                candidate = key
        if candidate is None and self._nanlike(old_label):
            # 若有多個 NaN/壞索引，請明確指定其值
            nan_idx = list(df.index[pd.to_datetime(df.index, errors="coerce").isna()])
            if len(nan_idx) == 1:
                candidate = nan_idx[0]
            elif len(nan_idx) > 1:
                raise KeyError(f"存在多個無效索引，請明確指定；候選：{nan_idx[:5]} ...")
        if candidate is None:
            raise KeyError(f"找不到舊索引：{old_label!r}")

        if (not overwrite) and (new_date_str in df.index):
            raise KeyError(f"新索引已存在：{new_date_str}（若要覆蓋，請設 overwrite=True）")

        row = df.loc[[candidate]].copy()
        df2 = df.drop(index=[candidate])
        row.index = pd.Index([new_date_str])
        out = pd.concat([df2, row]).sort_index()
        if in_place:
            self._df = out
        self.logger.info(f"[log] 索引改名：{candidate!r} → {new_date_str}")
        return out
# === end ===



def scan_pkl_tree(base: Path, layer: str = "", suffix: str = ".pkl") -> pd.DataFrame:
    """
    掃描某個根目錄底下的 pkl 檔，假設結構為：
        base/
          <item>/
            <subitem>.pkl
            ...

    參數
    ----
    base : Path
        根目錄，例如 dbpath_cleaned 或 dbpath_source。
    layer : str
        標記來源層級，例如 'cleaned' 或 'source'，方便後續 pivot。
    suffix : str
        要掃描的副檔名（預設為 '.pkl'）。

    回傳
    ----
    DataFrame，欄位：
        - layer
        - item
        - subitem
        - path
        - size_mb
        - mtime  (datetime)
    """
    base = Path(base)
    rows = []
    if not base.exists():
        return pd.DataFrame(columns=["layer", "item", "subitem", "path", "size_mb", "mtime"])

    for item_dir in base.iterdir():
        if not item_dir.is_dir():
            continue
        item = item_dir.name
        for p in item_dir.rglob(f"*{suffix}"):
            if not p.is_file():
                continue
            stat = p.stat()
            rows.append(
                {
                    "layer": layer,
                    "item": item,
                    "subitem": p.stem,
                    "path": str(p),
                    "size_mb": stat.st_size / 1024 / 1024,
                    "mtime": _dt.datetime.fromtimestamp(stat.st_mtime),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["layer", "item", "subitem", "path", "size_mb", "mtime"])
    df = pd.DataFrame(rows)
    df["mtime"] = pd.to_datetime(df["mtime"])
    return df

def add_days_lag(
    df: pd.DataFrame,
    date_col: str = "mtime",
    today: _dt.date | None = None,
) -> pd.DataFrame:
    """
    根據日期欄位計算與今天的落後天數，回傳新的 DataFrame（不修改原始 df）。

    會新增：
        - days_lag : int/float，today - date_col（天數）
    """
    if df.empty or date_col not in df.columns:
        return df.copy()

    out = df.copy()

    # today 預設為今天
    if today is None:
        today = _dt.date.today()

    # 轉成 datetime64[ns]
    dates = pd.to_datetime(out[date_col])

    # 差值會是 timedelta64[ns]，用 .dt.days 取天數
    today_ts = pd.to_datetime(today)
    delta = today_ts - dates
    out["days_lag"] = delta.dt.days  # 不再做 astype("timedelta64[D]")

    return out



def add_status_by_lag(
    df: pd.DataFrame,
    lag_col: str = "days_lag",
    ok_threshold: int = 1,
    warn_threshold: int = 5,
    col_name: str = "status",
) -> pd.DataFrame:
    """
    根據 days_lag 給出狀態標記：
        - <= ok_threshold        → 'OK'
        - <= warn_threshold      → 'LAGGING'
        - > warn_threshold 或 NA → 'STALE'
    """
    if df.empty:
        return df.copy()
    out = df.copy()

    def _status(v):
        try:
            v_int = int(v)
        except Exception:
            return "STALE"
        if v_int <= ok_threshold:
            return "OK"
        elif v_int <= warn_threshold:
            return "LAGGING"
        else:
            return "STALE"

    out[col_name] = out.get(lag_col, pd.Series([None] * len(out))).map(_status)
    return out


if __name__ == "__main__":
    logmanager = LogMaintainer()
    log = logmanager.df()
    invalid = logmanager.list_invalid_index()
    to_drop = invalid["label"].tolist()[:]  # 例如先刪前 10 個
    logmanager.delete_labels(to_drop, in_place=True)
    logmanager.save()