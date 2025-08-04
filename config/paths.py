from os.path import join
from config.conf import collection
from StevenTricks.file_utils import PathWalk_df
from pathlib import Path

# 指定warehouse(倉庫)的路徑
path_dic = {
    "stock_twse_db": {
        "db": r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse",
        "cache": r"/Users/stevenhsu/Documents/stock/twse"
    }
}

dbpath = path_dic["stock_twse_db"]["db"]
dbpath_source = join(dbpath, "source")
dbpath_cleaned = join(dbpath, "cleaned")
dbpath_cleaned_log = join(dbpath_cleaned, "log.pkl")
dbpath_log = join(dbpath_source, "log", "log.pkl")
dbpath_errorlog = join(dbpath_source, "log", "errorlog.pkl")
dbpath_productlist = join(dbpath_source, "productlist.pkl")

cachepath = path_dic["stock_twse_db"]["cache"]
cachepath_namelog = join(cachepath, "namelog.pkl")

dbpath_cleaned_db_df = PathWalk_df(dbpath_cleaned, fileinclude=[".pkl"], level=2)

cleaned_db_dict = {}

for key in collection:
    # 改為根據父資料夾名稱（即資料夾名稱等於 key）
    paths = dbpath_cleaned_db_df.loc[dbpath_cleaned_db_df["dir"] == key, "path"].tolist()

    if len(paths) >= 1:
        # 如果資料夾下有多個 .pkl 檔案，只要挑其中一個檔案的路徑，取其上層資料夾即可
        cleaned_db_dict[key] = str(Path(paths[0]).parent)
    else:
        raise ValueError(f"Error: Expected at least 1 file under folder '{key}', but found 0.")
