from pathlib import Path
from StevenTricks.staging import staging_path

# 指定 warehouse 路徑
path_dic = {
    "stock_twse_db": {
        "db_cloud": Path("/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse"),
        "db_local": Path("/Users/stevenhsu/programming/investment"),
        "cache": Path("/Users/stevenhsu/Documents/stock/twse")
    }
}


db_root = path_dic["stock_twse_db"]["db_cloud"],
cache_root = path_dic["stock_twse_db"]["cache"]

# 資料路徑整理
dbpath_source = db_root / "source"
dbpath_cleaned = db_root / "cleaned"
dbpath_cleaned_log = dbpath_cleaned / "log.pkl"
dbpath_log = dbpath_source / "log" / "log.pkl"
dbpath_errorlog = dbpath_source / "log" / "errorlog.pkl"
dbpath_productlist = dbpath_source / "productlist.pkl"

cachepath_namelog = cache_root / "namelog.pkl"

