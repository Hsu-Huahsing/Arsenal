from os.path import join


#因為爬蟲的結果會有很多db要儲存，要先指定warehouse(倉庫)的路徑
path_dic = {
    "stock_twse_db": {
        "db":r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse",
        "cache":r"/Users/stevenhsu/Documents/stock/twse"
    }
}

dbpath = path_dic["stock_twse_db"]["db"]
dbpath_source = join(dbpath, "source")
dbpath_cleaned = join(dbpath, "cleaned")
dbpath_cleaned_log = join(dbpath_cleaned,"log.pkl")
dbpath_log = join(dbpath_source, "log", "log.pkl")
dbpath_errorlog = join(dbpath_source, "log", "errorlog.pkl")
dbpath_productlist = join(dbpath_source, "productlist.pkl")

cachepath = path_dic["stock_twse_db"]["cache"]