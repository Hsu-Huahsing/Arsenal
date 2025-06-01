#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:22:32 2020

@author: mac
"""

from StevenTricks.dfi import periodictable, DataFrameMerger
from StevenTricks.file_utils import logfromfolder,  picklesave, pickleload, sweep_path, PathWalk_df
from StevenTricks.process import sleepteller
from conf import collection, dailycollection, colname_dic, product_clean,dbpath,dbpath_source,dbpath_log,dbpath_errorlog,dbpath_productlist
from schema_utils import warehouseinit
from crawler.product_list import product_list
from crawler.stock import main

import pandas as pd
import datetime

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
        if datetime.date.today().isoformat() not in log.index:
            print("{} not in log index, updating the log table".format(str(datetime.date.today())))
            # log.loc[log.index=="1999-07-25",:]
            # log.index = pd.to_datetime(log.index)
            # for i in log.index:
            #     print(i)
            #     if i == "Store":continue
            #     pd.to_datetime(i)
            # a=log.loc[log.index.isin(["Store","ass"]),:]
            # log.loc[log.index==pd.to_datetime("1990-01-31"),"ddd"]="dddd"
            # log.loc["ass"]
            # log["ddd"].unique()
            # type(log.index[2])
            latest_log = periodictable(collection, datemin=max(pd.to_datetime(log.index))+datetime.timedelta(days=1))
            # 從上一次創建log的最新天數開始，所以要加一天，然後開始創建新的table
            log = pd.concat([log, latest_log])
        print("LOG讀取成功")
    else:
        log = periodictable(collection)
        log.index = log.index.astype(str)
        print("LOG重置成功")
    # 不管有沒有log，在爬蟲啟動之前都會根據目前資料夾的資料來更新log，確保抓取沒有遺漏
    # 再判斷有沒有errorlog
    if errorlog_info["exists"] is True:
        errorlog = pickleload(dbpath_errorlog)
    else:
        errorlog = pd.DataFrame()
    print("開始更新LOG")
    # 先盤點資料
    dbpath_list = PathWalk_df(dbpath_source, [], ["log"], [".DS"], [])
    # 再更新log檔
    log = logfromfolder(dbpath_list, log=log, fillval='succeed')
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
        product_df = product_df.rename(columns=colname_dic)
        product_df = product_df.replace({"\u3000": ""}, regex=True)
        if "ISINCode" in product_df:
            product_df["代號"] = product_df["ISINCode"].str.slice(product_clean["code"][key][0],product_clean["code"][key][1])
        product_dic[key] = product_df

    product = pd.concat(list(product_dic.values()),axis=0)
    if productlist_info["exists"] is False:
        picklesave(product, dbpath_productlist)
    else:
        product_old = pickleload(dbpath_productlist)
        product_manager = DataFrameMerger(product_old)
        product_old = product_manager.renew(product)
        picklesave(product_old, dbpath_productlist)

    main(
        collection=collection,
        log=log,
        errorlog=errorlog,
        dbpath_source=dbpath_source,
        dbpath_log=dbpath_log,
        dbpath_errorlog=dbpath_errorlog
    )

