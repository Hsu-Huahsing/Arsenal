from StevenTricks.fileop import pickleload, picklesave
from conf import collection, path_dic

from StevenTricks.dfi import periodictable

from os.path import exists,join
from datetime import datetime, timedelta
import pandas as pd


def path_get(db_name=""):
    return {"source_dir": join(path_dic[db_name], "source"),
            "source_log": join(path_dic[db_name], "source/log.pkl")}


if __name__ == "__main__":
    dbpath_dic = path_get(db_name="stock_twse_db")
    log_exist = exists(dbpath_dic["source_log"])

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
