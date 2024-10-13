from StevenTricks.fileop import pickleload, picklesave, runninginfo, logfromfolder
from StevenTricks.dfi import periodictable
from crawler.log import path_get
from conf import collection
from os.path import exists, join
from datetime import datetime, timedelta

import pandas as pd

if __name__ == '__main__':
    # 先把log處理好
    # 先讀取log檔案本身
    runninginfo()
    print("LOG處理程序開始")
    dbpath_dic = path_get(db_name="stock_twse_db")
    log_exist = exists(dbpath_dic["source_log"])
    # 簡單判斷log是否存在
    if log_exist is True:
        print("LOG不存在，即將生成新LOG")
        source_log = pickleload(dbpath_dic["source_log"])
        if str(datetime.today().date()) not in source_log.index :
            print("{} not in log index, updating the log table".format(str(datetime.today().date())))
            latestlog = periodictable(collection, datemin=source_log.index.max()+timedelta(days=1))
            # 從上一次創建log的最新天數開始，所以要加一天，然後開始創建新的table
            source_log = pd.concat([source_log, latestlog])

    elif log_exist is False:
        print("LOG存在，讀取LOG")
        source_log = periodictable(collection)
    print("LOG讀取成功")
    print("開始更新LOG")
    # 開始更新log檔
    source_log = logfromfolder(dbpath_dic["source_dir"], fileinclude=['.pkl'], fileexclude=['log'], direxclude=['stocklist'], dirinclude=[], log=source_log, fillval='succeed')
    # 比對資料夾內的資料，依照現有存在的資料去比對比較準確，有可能上次抓完，中間有動到資料
    print("Log檔案更新結束/nLog程序處理結束")
    # log處理結束

    print("開始進行網路資料下載")







