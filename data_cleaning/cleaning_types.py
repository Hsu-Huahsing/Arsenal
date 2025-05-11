
from data_cleaning.cleaning_utils import (
    convert_date_column,
    rename_columns_batch
)
import pandas as pd
from conf import colname_dic, dropcol, numericol
from copy import deepcopy
from schema_utils import safe_frameup_data


# 處理有groups的情況


def type3(data_dict):
    df = deepcopy(data_dict["data_cleaned_pre"])
    df = split_column_parentheses(df, keep_right=True, right_colname="unit")
    df = df.dropna()
    df = type1(data_dict)
    data_dict["data_cleaned"] = df
    return data_dict


def type4(data_dict):
    df = deepcopy(data_dict["data_cleaned_pre"])
    df = type1(data_dict)
    df = rename_columns_batch(df, [
        ("買進", "融券買進"),
        ("融券買進", "融資買進"),
        ("賣出", "融券賣出"),
        ("融券賣出", "融資賣出"),
        ("今日餘額", "今日融券餘額"),
        ("今日融券餘額", "今日融資餘額"),
        ("限額", "融券限額"),
        ("融券限額", "融資限額")
    ])
    data_dict["data_cleaned"] = df
    return data_dict


def type5(data_dict):
    df = deepcopy(data_dict["data_cleaned_pre"])
    df = type1(data_dict)
    df = convert_date_column(df, ['date'])
    data_dict["data_cleaned"] = df
    return data_dict


def type6(data_dict):
    df = deepcopy(data_dict["data_cleaned_pre"])
    df = type1(data_dict)
    df = rename_columns_batch(df, [
        ("前日餘額", "借券前日餘額"),
        ("借券前日餘額", "融券前日餘額")
    ])
    data_dict["data_cleaned"] = df
    return data_dict


def type7(data_dict):
    df = deepcopy(data_dict["data_cleaned_pre"])
    df.columns = [c.replace("</br>", "") for c in df.columns]
    df = type1(data_dict)
    data_dict["data_cleaned"] = df
    return data_dict


def type8(data_dict):
    df = deepcopy(data_dict["data_cleaned_pre"])
    df = type1(data_dict)
    df = convert_date_column(df, ['最近一次上市公司申報外資持股異動日期'])
    data_dict["data_cleaned"] = df
    return data_dict
