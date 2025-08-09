#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:22:32 2020

@author: mac
"""
from os.path import exists
from StevenTricks.df_utils import periodictable
from StevenTricks.file_utils import logfromfolder,  pickleio, PathWalk_df
from config.conf import collection
from config.paths import db_root,dbpath_source,dbpath_log,dbpath_errorlog
from schema_utils import warehouseinit
from crawler.stock import main
import pandas as pd
import datetime

if __name__ == "__main__":
    warehouseinit(db_root)
    # 每一次被當成主要模組呼叫，都會自動生成倉庫資料夾，為了確保一定有資料夾，所以每次使用都要呼叫一次
    # 先判斷有沒有log
    if exists(dbpath_log) is True:
        log = pickleio(dbpath_log,mode="load")
        # 有log還要判斷是不是最新的
        if datetime.date.today().isoformat() not in log.index:
            print("{} not in log index, updating the log table".format(str(datetime.date.today())))
            latest_log = periodictable(collection, datemin=max(pd.to_datetime(log.index, errors='coerce'))+datetime.timedelta(days=1))
            # 從上一次創建log的最新天數開始，所以要加一天，然後開始創建新的table
            log = pd.concat([log, latest_log])
        print("LOG讀取成功")
    else:
        log = periodictable(collection)
        log.index = log.index.strftime("%Y-%m-%d")
        print("LOG重置成功")
    # 不管有沒有log，在爬蟲啟動之前都會根據目前資料夾的資料來更新log，確保抓取沒有遺漏
    # 再判斷有沒有errorlog
    if exists(dbpath_errorlog) is True:
        errorlog = pickleio(dbpath_errorlog,mode="load")
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

    main(
        collection=collection,
        log=log,
        errorlog=errorlog,
        dbpath_source=dbpath_source,
        dbpath_log=dbpath_log,
        dbpath_errorlog=dbpath_errorlog
    )

