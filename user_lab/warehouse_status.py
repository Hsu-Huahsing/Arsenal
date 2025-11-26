"""
user_lab/warehouse_status_lab.py

目的：
    給你一支「可以直接打開 .py 檔互動」的倉庫管理員腳本。

特色：
    - 放在 user_lab（使用者互動區）
    - 不用 argparse、不靠 terminal 問答
    - 執行後會產生一組 DataFrame 變數在檔案 global scope：
        - today
        - cleaned_detail
        - item_summary
        - relation_status
        - missing
        - orphan
        - log_summary
        - error_log

    你可以：
        - 在 PyCharm / VSCode 直接 run，看下方變數視窗
        - 或在 IPython / Jupyter 用 `%run -i warehouse_status_lab.py` 保留變數
        - 然後任意對這些 DataFrame 做 filter / plot / export

使用方式：
    1. 確認這支檔案在 Arsenal/user_lab 底下。
    2. 在專案根目錄下執行：
        - python -m Arsenal.user_lab.warehouse_status_lab
       或者在 IDE 直接 run 這支檔案。
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# 嘗試用 __file__ 判斷專案 root；若在 IPython / notebook 沒有 __file__，
# 就假設目前工作目錄 (cwd) 就是 Arsenal 專案 root。
# ---------------------------------------------------------------------------

if "__file__" in globals():
    # 正常以 .py 檔執行（python -m 或 IDE run）
    _THIS_FILE = Path(__file__).resolve()      # .../Arsenal/user_lab/warehouse_status_lab.py
    _PROJECT_ROOT = _THIS_FILE.parents[1]      # .../Arsenal
else:
    # 在 IPython / Jupyter 直接貼 code 執行時沒有 __file__
    # 這裡假設你是從 Arsenal 專案根目錄啟動的 notebook：
    #   cwd = /Users/.../Arsenal
    _PROJECT_ROOT = Path.cwd()

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

# 若 StevenTricks 與 Arsenal 在同一層 (/some/path/Arsenal, /some/path/StevenTricks)，則自動加入
_ST_ROOT = _PROJECT_ROOT.parent / "StevenTricks"
if _ST_ROOT.exists() and str(_ST_ROOT) not in sys.path:
    sys.path.append(str(_ST_ROOT))



# ---------------------------------------------------------------------------
# 二、匯入倉庫引擎
# ---------------------------------------------------------------------------

from warehouse_manager.twse_inventory import get_twse_status


# ---------------------------------------------------------------------------
# 三、可調的「互動參數區」
#    之後你要改顯示行為，優先改這邊就好。
# ---------------------------------------------------------------------------

SHOW_LOG = True       # 是否載入 & 使用 log_summary / error_log
MAX_ROWS = 30         # print 出來時，明細最多顯示幾列
SHOW_DETAIL = False   # 是否額外印出 cleaned_detail / relation_status 前幾列


# ---------------------------------------------------------------------------
# 四、核心呼叫：取得倉庫現況（這裡才是真的在動）
# ---------------------------------------------------------------------------
def _datetime_to_date_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    """
    把指定欄位（若存在且為 datetime）改成只保留日期（YYYY-MM-DD）。
    會直接修改傳入的 df。
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date

status = get_twse_status(include_log=SHOW_LOG, include_errorlog=SHOW_LOG)

# 方便互動使用的變數拆解
today: pd.Timestamp | None = status["today"]
cleaned_detail: pd.DataFrame = status["cleaned_detail"]
item_summary: pd.DataFrame = status["item_summary"]
relation_status: pd.DataFrame = status["relation_status"]
missing: pd.DataFrame = status["missing"]
orphan: pd.DataFrame = status["orphan"]
log_summary: pd.DataFrame = status["log_summary"]
error_log: pd.DataFrame = status["error_log"]

# 把部分欄位的時間壓成 date（避免 print 出一大串秒數）
_datetime_to_date_inplace(item_summary, ["min_mtime", "max_mtime"])
_datetime_to_date_inplace(missing, ["source_mtime", "cleaned_mtime"])
_datetime_to_date_inplace(orphan, ["source_mtime", "cleaned_mtime"])

# ---------------------------------------------------------------------------
# 五、基本摘要輸出（讓你用 script 跑，也能一眼看狀況）
# ---------------------------------------------------------------------------

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

def _title(text: str) -> str:
    line = "-" * len(text)
    return f"{text}\n{line}"

print(_title(f"TWSE 倉庫總覽（截至 {today}）"))

# 1. item 層級 summary
ITEM_MAX_ROWS = 50  # 可以拉到最上面跟 SHOW_LOG 一起當參數

print("\n[1] Cleaned 依 item 彙總（依 max_days_lag 由大到小，僅顯示前幾項）：")
if item_summary.empty:
    print("  （尚無 cleaned 資料）")
else:
    # 只挑主要欄位，避免整張表太寬
    cols_prefer = [
        "item",
        "n_subitems",
        "n_files",
        "min_mtime",
        "max_mtime",
        "max_days_lag",
        "n_ok",
        "n_lagging",
        "n_stale",
    ]
    cols_exist = [c for c in cols_prefer if c in item_summary.columns]

    display_item = item_summary.sort_values(
        "max_days_lag",
        ascending=False,
        na_position="last",
    )[cols_exist]

    print(display_item.head(ITEM_MAX_ROWS).to_string(index=False))
    if len(display_item) > ITEM_MAX_ROWS:
        print(f"\n  ... 共 {len(display_item)} 個 item，只顯示前 {ITEM_MAX_ROWS} 個。")


# 2. source vs cleaned 缺漏
print("\n[2] Source vs Cleaned 缺漏檢查：")
if missing.empty and orphan.empty:
    print("  ✔ source / cleaned 結構一致，沒有缺漏或孤兒檔。")
else:
    if not missing.empty:
        print("\n  2.1 source 有但 cleaned 沒有（缺少清理或路徑設定）：")
        cols = [c for c in ["item", "subitem", "source_mtime", "source_status"] if c in missing.columns]
        print(missing[cols].head(MAX_ROWS).to_string(index=False))
        if len(missing) > MAX_ROWS:
            print(f"  ... 共 {len(missing)} 筆，只顯示前 {MAX_ROWS} 筆。")

    if not orphan.empty:
        print("\n  2.2 cleaned 有但 source 沒有（可能為舊格式或 source 路徑未設定）：")
        cols = [c for c in ["item", "subitem", "cleaned_mtime", "cleaned_status"] if c in orphan.columns]
        print(orphan[cols].head(MAX_ROWS).to_string(index=False))
        if len(orphan) > MAX_ROWS:
            print(f"  ... 共 {len(orphan)} 筆，只顯示前 {MAX_ROWS} 筆。")


# 3. log 摘要（選配）
if SHOW_LOG:
    print("\n[3] 清理 log 摘要（若 log.pkl 存在）：")
    if log_summary.empty:
        print("  （找不到 log.pkl 或內容為空）")
    else:
        print(log_summary.head(MAX_ROWS).to_string(index=False))
        if len(log_summary) > MAX_ROWS:
            print(f"  ... 共 {len(log_summary)} 筆，只顯示前 {MAX_ROWS} 筆。")

    print("\n[4] error log 摘要（若 errorlog.pkl 存在）：")
    if error_log.empty:
        print("  （找不到 errorlog.pkl 或內容為空）")
    else:
        print("  欄位：", ", ".join(map(str, error_log.columns)))
        print("  內容（前幾列）：")
        print(error_log.head(MAX_ROWS).to_string(index=False))


# ---------------------------------------------------------------------------
# 六、互動示範（不會自動執行，你要時自己拿來改）
# ---------------------------------------------------------------------------

if False:  # ← 改成 True 就會在執行時跑這段示範
    # 範例：只看 lag 很大的 cleaned 檔案
    lag_threshold = 5
    big_lag = cleaned_detail.loc[cleaned_detail["days_lag"] > lag_threshold]
    print(f"\n[Demo] days_lag > {lag_threshold} 的 cleaned 檔案（前 {MAX_ROWS} 筆）：")
    print(big_lag.head(MAX_ROWS).to_string(index=False))