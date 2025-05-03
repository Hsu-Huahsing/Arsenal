# -*- coding: utf-8 -*-
""" Schema Utilities
處理資料結構分析與欄位重建，如多層 dict 拆解、欄位命名分類、表格重建等。
依賴 StevenTricks 中的欄位定義模組（colname_dic, numericol 等）。
"""

import pandas as pd
from os import makedirs
from pathlib import Path

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