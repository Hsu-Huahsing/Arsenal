
from data_cleaning.cleaning_utils import (
    split_column_parentheses,
    convert_date_column,
    rename_columns_batch,
    safe_numeric_convert
)
import pandas as pd
from conf import colname_dic, dropcol, numericol


def type1(df, title, subtitle, file_name):
    df = df.replace({",": "", r'\)': '', r'\(': '_'}, regex=True)
    df = df.rename(columns=colname_dic)
    df = df.drop(columns=dropcol, errors='ignore')
    cols = numericol[title][subtitle]
    df = safe_numeric_convert(df, cols)
    return {
        "title" : title,
        "subtitle" : subtitle,
        "file_name" : file_name,
        "data" : df
    }


def type2(df, title, subtitle, file_name):
    df_left = split_column_parentheses(df)
    df_right = df.loc[:, df.columns].str.extract(r'\(([^)]*)\)')
    df_right.columns = df.columns
    df = pd.concat([df_left, df_right.dropna(how='all')], ignore_index=True)
    df = type1(df, title, subtitle)[subtitle]
    return {
        "title" : title,
        "subtitle" : subtitle,
        "file_name" : file_name,
        "data" : df
    }


def type3(df, title, subtitle, file_name):
    df = split_column_parentheses(df, keep_right=True, right_colname="unit")
    df = df.dropna()
    df = type1(df, title, subtitle)[subtitle]
    return {
        "title" : title,
        "subtitle" : subtitle,
        "file_name" : file_name,
        "data" : df
    }


def type4(df, title, subtitle, file_name):
    df = type1(df, title, subtitle)[subtitle]
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
    return {
        "title" : title,
        "subtitle" : subtitle,
        "file_name" : file_name,
        "data" : df
    }


def type5(df, title, subtitle, file_name):
    df = type1(df, title, subtitle)[subtitle]
    df = convert_date_column(df, ['date'])
    return {
        "title" : title,
        "subtitle" : subtitle,
        "file_name" : file_name,
        "data" : df
    }


def type6(df, title, subtitle, file_name):
    df = type1(df, title, subtitle)[subtitle]
    df = rename_columns_batch(df, [
        ("前日餘額", "借券前日餘額"),
        ("借券前日餘額", "融券前日餘額")
    ])
    return {
        "title" : title,
        "subtitle" : subtitle,
        "file_name" : file_name,
        "data" : df
    }


def type7(df, title, subtitle, file_name):
    df.columns = [c.replace("</br>", "") for c in df.columns]
    df = type1(df, title, subtitle)[subtitle]
    return {
        "title" : title,
        "subtitle" : subtitle,
        "file_name" : file_name,
        "data" : df
    }


def type8(df, title, subtitle, file_name):
    df = type1(df, title, subtitle)[subtitle]
    df = convert_date_column(df, ['最近一次上市公司申報外資持股異動日期'])
    return {
        "title" : title,
        "subtitle" : subtitle,
        "file_name" : file_name,
        "data" : df
    }
