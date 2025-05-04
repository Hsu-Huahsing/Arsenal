#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:22:32 2020

@author: mac
"""

from StevenTricks.dfi import findval, periodictable, DataFrameMerger
from StevenTricks.netGEN import randomheader
from StevenTricks.file_utils import logfromfolder, logmaker, picklesave, pickleload, sweep_path, PathWalk_df
from StevenTricks.process import sleepteller
from conf import collection, dailycollection, path_dic, product_col, product_clean
from crawler.schema_utils import warehouseinit
from crawler.product_list import product_list
from crawler.run_crawler_task import main
from os import remove, makedirs
from os.path import join, exists
from traceback import format_exc

import requests as re
import sys
import pandas as pd
import datetime

dbpath = path_dic["stock_twse_db"]
dbpath_source = join(dbpath, "source")
dbpath_log = join(dbpath_source, "log", "log.pkl")
dbpath_errorlog = join(dbpath_source, "log", "errorlog.pkl")
dbpath_productlist = join(dbpath_source, "productlist.pkl")
log_info = sweep_path(dbpath_log)
errorlog_info = sweep_path(dbpath_errorlog)
productlist_info = sweep_path(dbpath_productlist)

if __name__ == "__main__":
    warehouseinit(dbpath)
    # 每一次被當成主要模組呼叫，都會自動生成倉庫資料夾，為了確保一定有資料夾，所以每次使用都要呼叫一次
    # 先判斷有沒有log
    if log_info["exists"] is True:
        log = pickleload(dbpath_log)
        # 有log還要判斷是不是最新的
        if datetime.date.today() not in log.index:
            print("{} not in log index, updating the log table".format(str(datetime.date.today())))
            latest_log = periodictable(collection, datemin=log.index.max()+datetime.timedelta(days=1))
            # 從上一次創建log的最新天數開始，所以要加一天，然後開始創建新的table
            log = pd.concat([log, latest_log])
    else:
        log = periodictable(collection)
    # 不管有沒有log，在爬蟲啟動之前都會根據目前資料夾的資料來更新log，確保抓取沒有遺漏
    # 再判斷有沒有errorlog
    if errorlog_info["exists"] is True:
        errorlog = pickleload(dbpath_errorlog)
    else:
        errorlog = pd.DataFrame()
    print("LOG讀取成功")
    print("開始更新LOG")
    # 先盤點資料
    dbpath_lis = PathWalk_df(dbpath_source, [], ["log"], [], [])
    # 再更新log檔
    log = logfromfolder(dbpath_lis, log=log, fillval='succeed')
    # 比對資料夾內的資料，依照現有存在的資料去比對比較準確，有可能上次抓完，中間有動到資料
    print("Log檔案更新結束\nLog程序處理結束")
    # log處理結束
    print("開始進行網路資料下載")
    product_dic = {}
    # 先進行商品清單下載
    for _ in dailycollection['stocklist']['modelis']:
        product = product_list(dailycollection['stocklist']['url'].format(str(_)))
        product_dic[_] = product
        print(_)
        sleepteller()
    # 針對商品清單做資料清理
    for key in product_dic:
        product_df = product_dic[key]
        product_df = product_df.rename(columns=product_col)
        product_df = product_df.replace({"\u3000": ""}, regex=True)
        if "國際證券辨識號碼(ISIN Code)" in product_df:
            product_df["代號"] = product_df["國際證券辨識號碼(ISIN Code)"].str.slice(product_clean["code"][key][0],product_clean["code"][key][1])
        product_dic[key] = product_df

    product = pd.concat(list(product_dic.values()),axis=0)
    if productlist_info["exists"] is False:
        picklesave(product, dbpath_productlist)
    else:
        product_old = pickleload(dbpath_productlist)
        product_manager = DataFrameMerger(product_old)
        product_old = product_manager.renew(product)

    main(
        collection=collection,
        log=log,
        errorlog=errorlog,
        dbpath_source=dbpath_source,
        dbpath_log=dbpath_log,
        dbpath_errorlog=dbpath_errorlog
    )

