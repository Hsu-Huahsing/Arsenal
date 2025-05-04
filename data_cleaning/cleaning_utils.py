
import pandas as pd


def split_column_parentheses(df, keep_right=False, right_colname="unit"):
    """將每欄資料以 '(' 拆分成左（主）與右（副），副欄位名稱可自訂。"""
    res = []
    for subcol in df.columns:
        temp = df[subcol].str.split(r'\(', expand=True)
        if keep_right:
            temp.columns = [subcol, right_colname]
        else:
            temp.columns = [subcol]
        res.append(temp)
    result = pd.concat(res, axis=1)
    return result


def convert_date_column(df, cols, mode=4):
    """將指定欄位轉為 datetime 格式（可指定格式處理邏輯）"""
    df[cols] = df[cols].apply(pd.to_datetime, errors='coerce')
    return df


def rename_columns_batch(df, replace_pairs):
    """依照替換規則（old, new）依序套用第一個匹配欄位"""
    colstr = ",".join(df.columns)
    for old, new in replace_pairs:
        colstr = colstr.replace(old, new, 1)
    df.columns = colstr.split(",")
    return df


def safe_numeric_convert(df, cols):
    """將指定欄位轉成數值型態，無法轉換設為 NaN"""
    cols = [_ for _ in cols if _ in df]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df
