# Arsenal/warehouse/warehouse_manager.py

from pathlib import Path
import pandas as pd
from datetime import datetime
from config.paths import dbpath_cleaned, dbpath_source, dbpath_log, dbpath_errorlog

def scan_cleaned_files() -> pd.DataFrame:
    """
    掃描 cleaned DB 資料夾，列出所有 item / subitem 對應的檔案、大小與最後更新時間。
    """
    rows = []
    base = Path(dbpath_cleaned)

    for item_dir in base.iterdir():
        if not item_dir.is_dir():
            continue
        item = item_dir.name
        for p in item_dir.glob("*.pkl"):
            subitem = p.stem
            stat = p.stat()
            rows.append(
                {
                    "layer": "cleaned",
                    "item": item,
                    "subitem": subitem,
                    "path": str(p),
                    "size_mb": stat.st_size / 1024 / 1024,
                    "mtime": datetime.fromtimestamp(stat.st_mtime),
                }
            )
    return pd.DataFrame(rows)


def scan_source_files() -> pd.DataFrame:
    """
    類似方法掃描 source DB（原始爬回來的），方便對照「source 有、cleaned 沒有」的缺口。
    """
    rows = []
    base = Path(dbpath_source)

    for item_dir in base.iterdir():
        if not item_dir.is_dir():
            continue
        item = item_dir.name
        for p in item_dir.rglob("*.pkl"):
            subitem = p.stem
            stat = p.stat()
            rows.append(
                {
                    "layer": "source",
                    "item": item,
                    "subitem": subitem,
                    "path": str(p),
                    "size_mb": stat.st_size / 1024 / 1024,
                    "mtime": datetime.fromtimestamp(stat.st_mtime),
                }
            )
    return pd.DataFrame(rows)


def summarize_inventory() -> pd.DataFrame:
    """
    把 source / cleaned 的掃描結果合併，整理成一張「倉庫總表」。
    """
    src = scan_source_files()
    cln = scan_cleaned_files()

    df = pd.concat([src, cln], ignore_index=True)

    # 這邊可以做一些 pivot / groupby：例如每個 item 在 source / cleaned 是否都存在
    summary = (
        df.pivot_table(
            index=["item", "subitem"],
            columns="layer",
            values=["size_mb", "mtime"],
            aggfunc="max",
        )
        .sort_index()
    )
    return summary


def quick_inventory_summary() -> pd.DataFrame:
    """
    給 User Zone 用的簡易入口：
    - 回傳每個 item / subitem 的 cleaned mtime，順便標記「是否落後今天超過 N 天」。
    """
    inv = scan_cleaned_files()
    if inv.empty:
        return inv

    today = datetime.today().date()
    inv["mtime_date"] = inv["mtime"].dt.date
    inv["days_lag"] = (today - inv["mtime_date"]).dt.days

    # 以 item/subitem 聚合：只看最新的一個檔案
    grouped = (
        inv.sort_values("mtime", ascending=False)
        .groupby(["item", "subitem"], as_index=False)
        .first()
    )

    # 加個簡單的健康狀態
    def status(row):
        if row["days_lag"] <= 1:
            return "OK"
        elif row["days_lag"] <= 5:
            return "LAGGING"
        else:
            return "STALE"

    grouped["status"] = grouped.apply(status, axis=1)
    return grouped[["item", "subitem", "mtime", "days_lag", "status", "size_mb", "path"]]
f
