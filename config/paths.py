from os.path import join
from config.conf import collection
from StevenTricks.file_utils import PathWalk_df

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

dbpath_cleaned_db_df = PathWalk_df(dbpath_cleaned, fileinclude=[".pkl"],level=2)

cleaned_db_dict = {}

for key in collection:
    paths = dbpath_cleaned_db_df.loc[dbpath_cleaned_db_df["file"] == f"{key}.db", "path"].tolist()

    if len(paths) == 1:
        cleaned_db_dict[key] = paths[0]
    else:
        raise ValueError(f"Error: Expected exactly 1 path for '{key}.db', but found {len(paths)} paths: {paths}")