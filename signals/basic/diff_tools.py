from config.paths import dbpath_cleaned
from config.col_format import numericol
from StevenTricks.internal_db import DBPkl

from StevenTricks.convert_utils import safe_numeric_convert
import pandas as pd
from typing import Sequence


def diff_previous(
    df: pd.DataFrame,
    sort_cols: Sequence[str] = ("代號", "date"),
    item: str = "",
    subitem: str = "",
    percent: bool = True,
) -> pd.DataFrame:
    """
    對同一「代號」的時間序列，計算「當日 - 前一日」的差值與百分比變化。

    參數：
        df          : 原始資料表，至少需要包含 sort_cols 以及用來分組的「代號」欄位。
        sort_cols   : 用來排序的欄位，預設為 ("代號", "date")，會用 descending 排序，
                      讓前一筆資料剛好是「上一個日期」。
        item        : numericol 設定中的第一層 key。
        subitem     : numericol 設定中的第二層 key。
        percent     : 若為 True，會另外計算 diff_percent_{col} = diff_{col} / previous_{col}。

    回傳：
        一份新的 DataFrame，包含：
            - 原始欄位
            - previous_{col}：前一日數值
            - diff_{col}    ：當日與前一日差值
            - diff_percent_{col}（若 percent=True）
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df 必須是 pandas.DataFrame")

    temp = df.copy()

    # 依 sort_cols 做排序（通常是 代號 + 日期 由新到舊）
    temp = temp.sort_values(by=list(sort_cols), ascending=False)

    # 依 numericol[item][subitem] 取得需要計算差異的欄位
    value_col = numericol[item][subitem]
    # 過濾掉現在 df 中沒有的欄位，避免 KeyError
    value_col = [c for c in value_col if c in temp.columns]

    # 將目標欄位轉成數值型態（內部會處理錯誤值）
    temp = safe_numeric_convert(temp, value_col)

    # 對每一個數值欄位，根據「代號」分組後做 shift(-1)，取得「前一日」資料
    for col in value_col:
        prev_col = f"previous_{col}"
        diff_col = f"diff_{col}"
        pct_col = f"diff_percent_{col}"

        temp[prev_col] = temp.groupby("代號")[col].shift(-1)
        temp[diff_col] = temp[col] - temp[prev_col]

        if percent:
            # 避免 division by zero 或 NaN，分母為 0 或 NaN 時結果設為 NaN
            temp[pct_col] = temp[diff_col] / temp[prev_col].replace({0: pd.NA})

    return temp




if __name__ == "__main__":
    # Demo 範例：實務上會由其他模組呼叫，而不是直接執行本檔

    # 1. 決定這個「DB 資料夾」的位置
    db_name = dbpath_cleaned / "三大法人買賣超日報"  # 注意：這裡已經不是 .db 檔，而是資料夾

    # 2. 這個 DB 底下的一張表（Arsenal twse 寫入時用的 subitem 名稱）
    table_name = "三大法人買賣超日報"  # 看 config.conf 裡的 subtitle，跟 cleaning 時用的 subitem 一致

    # 3. 建立 DBPkl 物件
    db = DBPkl(str(db_name), table_name)

    # 4. 讀取資料
    df = db.load_db(decode_links=True)  # decode_links=True：把 link_id 還原成原本的字串分類
    # df_raw = db.load_db(decode_links=False)  # 保留整數 ID，比較適合做純運算

    # 把 date 欄位從 object → datetime64[ns]，並同步更新 schema
    db.migrate_column_dtype("date", "datetime64[ns]")


    raw_df = db.load_raw()
    schema = db.load_schema()

    mismatches = []
    for col, expected in schema.get("dtypes", {}).items():
        if col not in raw_df.columns:
            print(f"[WARN] schema 有欄位 {col}，但 df 裡沒有，先略過")
            continue
        actual = str(raw_df[col].dtype)
        if actual != expected:
            mismatches.append((col, expected, actual))

    print("=== dtype 不一致欄位列表 ===")
    for col, expected, actual in mismatches:
        print(f"{col}: schema={expected}, df={actual}")


    for col, expected, actual in mismatches:
        # 只處理 schema=int, df=float 的 case
        if expected.startswith("int") and actual.startswith("float"):
            print(f"[MIGRATE] {col}: {expected} -> {actual}")
            db.migrate_column_dtype(col, actual)