import pandas as pd
from typing import  Optional, Tuple


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