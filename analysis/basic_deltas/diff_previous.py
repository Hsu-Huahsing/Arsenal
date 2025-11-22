from config.paths import dbpath_cleaned
from config.col_format import numericol
from StevenTricks.internal_db import DBPkl

from StevenTricks.convert_utils import safe_numeric_convert
from StevenTricks.df_utils import make_series
import pandas as pd
from typing import Sequence, Optional, Tuple


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


def cumulative_tracker(
    series: pd.Series,
    start_date,
    threshold: Optional[float] = None,
    end_date=None,
    direction: str = "forward",
) -> Tuple[Optional[pd.Timestamp], float, str]:
    """
    在時間序列中，從指定 start_date 開始累積 series 的值，
    直到達到指定 threshold 或到達 end_date 為止。

    參數：
        series    : index 為日期的數值序列。
        start_date: 起始日期（str 或 datetime）。
        threshold : 累積目標門檻，若為 None 則必須提供 end_date。
        end_date  : 結束日期上限（含），若為 None 則由資料自然結束。
        direction : 'forward' 代表往時間向後累積；
                    'backward' 代表從 start_date 往前累積。

    回傳：
        (停止時的日期, 累積值, 停止原因)
        停止原因為：
            - 'threshold'：累積值達到門檻
            - 'end_date'：到達 end_date（但未達門檻）
            - 'exhausted'：資料用盡
    """
    if threshold is None and end_date is None:
        raise ValueError("必須提供 threshold 或 end_date 其中之一，以避免無窮迴圈。")

    if not isinstance(series, pd.Series):
        raise TypeError("series 必須是 pandas.Series")

    # 嘗試將 index 視為日期
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if end_date is not None and isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # 依方向取出合適區間並排序
    if direction == "forward":
        data = series[series.index >= start_date].sort_index()
        if end_date is not None:
            data = data[data.index <= end_date]
    elif direction == "backward":
        data = series[series.index <= start_date].sort_index(ascending=False)
        if end_date is not None:
            data = data[data.index >= end_date]
    else:
        raise ValueError("direction 必須是 'forward' 或 'backward'。")

    cumulative = 0.0
    for date, value in data.items():
        # 跳過 NaN
        if pd.isna(value):
            continue
        cumulative += float(value)
        if threshold is not None and cumulative >= threshold:
            return date, cumulative, "threshold"

    # 沒有達到門檻，資料用完或被 end_date 截斷
    if not data.empty:
        if end_date is not None:
            return data.index[-1], cumulative, "end_date"
        else:
            return data.index[-1], cumulative, "exhausted"
    else:
        return None, 0.0, "exhausted"


class CumulativeMatcher:
    """
    輔助類別：
    1. 先用 set_reference() 設定一段「參考期間」，並計算該期間的累積值。
    2. 再用 find_matching_period()，從新的起點開始尋找「累積值相同」的期間。
    """
    def __init__(self, series: pd.Series):
        if not isinstance(series, pd.Series):
            raise TypeError("series 必須是 pandas.Series")
        if not isinstance(series.index, pd.DatetimeIndex):
            s = series.copy()
            s.index = pd.to_datetime(s.index)
            self.series = s.sort_index()
        else:
            self.series = series.sort_index()

        self.reference_start: Optional[pd.Timestamp] = None
        self.reference_end: Optional[pd.Timestamp] = None
        self.reference_sum: Optional[float] = None

    def set_reference(self, start_date, end_date) -> None:
        """
        設定一段參考區間，並計算該期間的累積值(reference_sum)。
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # 只取參考區間內的資料
        ref_series = self.series[(self.series.index >= start) & (self.series.index <= end)]

        if ref_series.empty:
            raise ValueError("參考區間內沒有任何資料。")

        # 累積總和
        cum_sum = float(ref_series.sum())

        self.reference_start = start
        self.reference_end = end
        self.reference_sum = cum_sum

    def find_matching_period(
        self,
        start_date=None,
        direction: str = "forward",
        threshold: Optional[float] = None,
        end_date=None,
    ):
        """
        以之前 set_reference() 算出的 reference_sum 當作門檻，或自行指定 threshold，
        從 start_date 開始往前或往後累積，尋找累積值達到同一門檻的日期。

        回傳與 cumulative_tracker 相同：
            (停止日期, 累積值, 停止原因)
        """
        # 若沒指定 threshold，就用參考區間的累積值
        if threshold is None:
            if self.reference_sum is None:
                raise ValueError("尚未設定參考區間，請先呼叫 set_reference()。")
            threshold = self.reference_sum

        # 若沒提供 start_date，就以參考區間的結束日期為起點
        if start_date is None:
            if self.reference_end is None:
                raise ValueError("必須提供 start_date 或先呼叫 set_reference()。")
            start_date = self.reference_end

        return cumulative_tracker(
            self.series,
            start_date=start_date,
            threshold=threshold,
            end_date=end_date,
            direction=direction,
        )


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