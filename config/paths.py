from pathlib import Path

# ---- 基本路徑設定 ----
# 之後如果你要換 iCloud 位置 / 本機 staging 根目錄，只要改這裡
path_dic = {
    "stock_twse_db": {
        # iCloud 上的正式 TWSE DB 根目錄
        "db_cloud": Path("/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse"),
        # 本機 staging 用的根目錄（只放暫存檔，不是正式 DB）
        "db_local": Path("/Users/stevenhsu/programming/investment"),
        # 產業金流等快取資料
        "cache": Path("/Users/stevenhsu/Documents/stock/twse"),
    },
}

# iCloud 正式 DB root
db_root: Path = path_dic["stock_twse_db"]["db_cloud"]
# 本機 staging root（會交給 staging_path 當 staging_root）
db_local_root: Path = path_dic["stock_twse_db"]["db_local"]
# cache root
cache_root: Path = path_dic["stock_twse_db"]["cache"]

# ---- TWSE 來源 / 清洗後路徑 ----

# crawler 抓回來的原始 pkl 目錄
dbpath_source: Path = db_root / "source"

# cleaning 後的 DB 目錄（DBPkl 資料庫 root）
dbpath_cleaned: Path = db_root / "cleaned"

# cleaning 完成紀錄 log（pickle）
dbpath_cleaned_log: Path = dbpath_cleaned / "log.pkl"

# crawler log / error log / productlist
dbpath_log: Path = dbpath_source / "log" / "log.pkl"
dbpath_errorlog: Path = dbpath_source / "log" / "errorlog.pkl"
dbpath_productlist: Path = dbpath_source / "productlist.pkl"

# ---- 產業金流 / 快取 ----

cachepath: Path = cache_root
cachepath_namelog: Path = cache_root / "namelog.pkl"

# 給 app.industry_flow 用的 mapping
cleaned_db_dict = {
    # DBPkl 的 root 資料夾，industry_flow 用 readsql_iter(dbpath=...) 讀
    "三大法人買賣超日報": dbpath_cleaned / "三大法人買賣超日報",
}
