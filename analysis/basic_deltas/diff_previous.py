from config.conf import dbpath_cleaned,numericol
from StevenTricks.dbsqlite import readsql_iter
from StevenTricks.convert_utils import safe_numeric_convert
from StevenTricks.df_utils import make_series
import pandas as pd


def diff_previous(df: pd.DataFrame, sort_col : ["代號","date"], item: str,subitem:str,percent=True):
    # 這是和前一天的資料去做相減看變化，所以一定要有date，如果資料沒有date，這個函式就沒有意義
    # 會返回一個有詳細計算步驟的temp，可以取代原本的df
    temp = df.copy()
    temp = temp.sort_values(by=sort_col, ascending=False)
    # 自動去抓數值類型的欄位，把這個項目底下數值的欄位全部去跟前一天相減，有新增欄位也要去cond.py新增
    value_col = numericol[item][subitem]
    print(value_col)
    value_col = [_ for _ in value_col if _ in temp]
    print(temp.columns)
    print(value_col)
    temp = safe_numeric_convert(temp, value_col)
    # 對 value_col 欄位，根據 "代號" 分組後做 shift
    for col in value_col:
        temp[f"previous_{col}"] = temp.groupby("代號")[col].shift(-1)
        temp[f'diff_{col}'] = temp[col] - temp[f"previous_{col}"]
        if percent is True:
            temp[f'diff_percent_{col}'] = temp[f'diff_{col}'] / temp[f'previous_{col}']

    return temp

def cumulative_tracker(
        series,
        start_date,
        threshold=None,
        end_date=None,
        direction="forward"
):
    """
    在時間序列中從指定日期累積值，直到達到指定門檻或到達終止日期為止。

    Parameters:
        series (pd.Series): 日期為 index，值為數值。
        start_date (str or datetime): 起始時間點。
        threshold (float or None): 累積的數值門檻。
        end_date (str or datetime or None): 最後允許的日期。
        direction (str): 累積方向，可選 "forward" 或 "backward"。

    Returns:
        tuple: (pd.Timestamp, 累積值, 停止原因)
            停止原因為 'threshold'（門檻達成）、
                          'end_date'（時間到）、
                          'exhausted'（資料用完）
    """
    if threshold is None and end_date is None:
        raise ValueError("Either 'threshold' or 'end_date' must be provided to avoid infinite loop.")

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # 排序 & 篩選方向
    if direction == "forward":
        data = series[series.index >= start_date].sort_index()
        if end_date is not None:
            data = data[data.index <= end_date]
    elif direction == "backward":
        data = series[series.index <= start_date].sort_index(ascending=False)
        if end_date is not None:
            data = data[data.index >= end_date]
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    cumulative = 0
    for date, value in data.items():
        cumulative += value
        if threshold is not None and cumulative >= threshold:
            return date, cumulative, 'threshold'

    # 沒達到門檻，資料走完
    if not data.empty:
        return data.index[-1], cumulative, 'end_date' if end_date else 'exhausted'
    else:
        return None, 0, 'exhausted'

class CumulativeMatcher:
    def __init__(self, series):
        self.series = series.sort_index()
        self.reference_start = None
        self.reference_end = None
        self.reference_sum = None

    def set_reference(self, start_date, end_date):
        """
        設定參考區間，透過 cumulative_tracker 累積 reference sum。
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        date, cum_sum, _ = cumulative_tracker(
            start_date=start,
            end_date=end,
            direction="forward"
        )
        self.reference_start = start
        self.reference_end = end
        self.reference_sum = cum_sum

    def find_matching_period(self, start_date=None, direction="forward", threshold=None, end_date=None):
        """
        根據參考值或指定門檻，在給定起點後尋找相同累積值的區段。

        Parameters:
            start_date (str or datetime): 搜尋起始點（預設為 reference_end）
            direction (str): "forward" 或 "backward"
            threshold (float or None): 要匹配的累積門檻，預設為 reference_sum
            end_date (str or datetime or None): 限制搜尋的最遠時間

        Returns:
            tuple: (pd.Timestamp, 累積值, 終止原因)
        """
        if threshold is None:
            if self.reference_sum is None:
                raise ValueError("No reference threshold defined.")
            threshold = self.reference_sum

        if start_date is None:
            if self.reference_end is None:
                raise ValueError("Must provide start_date or set reference first.")
            start_date = self.reference_end

        return cumulative_tracker(
            start_date=start_date,
            threshold=threshold,
            end_date=end_date,
            direction=direction
        )

if __name__ == "__main__":
    table1 = readsql_iter(dbpath=dbpath_cleaned,db_list=["外資及陸資投資持股統計.db"])
    table1 = next(table1)
    table2 = diff_previous(table1, ["代號", "date"], item="外資及陸資投資持股統計", subitem="外資及陸資投資持股統計")
    date_col = make_series(table2, "diff_全體外資及陸資持有股數")
    cumulative_tracker(date_col,"2023-1-1",)

