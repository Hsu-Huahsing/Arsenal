import pandas as pd
from data_cleaning.cleaning_utils import data_cleaned_df, data_cleaned_groups, frameup_safe,key_extract
from StevenTricks.dbsqlite import tosql_df
from conf import collection,dbpath_source,dbpath_cleaned,dbpath_cleaned_log,colname_dic,fields_span
from StevenTricks.file_utils import picklesave, pickleload, sweep_path, PathWalk_df
from StevenTricks.dictur import keyinstr
from os.path import join
from itertools import chain


log_info = sweep_path(dbpath_cleaned_log)

if __name__ == "__main__":
    dbpath_list = PathWalk_df(dbpath_source, [], ["log"], [], [".pkl"])
    if log_info["exists"] is True:
        log = pickleload(dbpath_cleaned_log)
        log_list = log.values.tolist()
        log_list = list(chain.from_iterable(log_list))
        # set(log_list)
        dbpath_list = dbpath_list.loc[~dbpath_list["file"].isin(log_list),:]
    else:
        log = pd.DataFrame()


    for file , path in dbpath_list[["file","path"]].values:
        print(file,path)
        if file == "productlist.pkl":
            productlist = pickleload(path)
            picklesave(productlist,join(dbpath_cleaned,file))
            continue
        file_info = sweep_path(path)
        file_data = pickleload(path)
        subtitle = file_data["crawlerdic"].get("subtitle",[file_info["parentdir"]])

        dict_list = key_extract(dic=file_data)
        dict_list_tables = []
        if "tables" in file_data:
            for table in file_data["tables"]:
                dict_list_tables += key_extract(dic=table)
        dict_list += dict_list_tables
        # 用抓table的方式，把固定的格式 title fields data groups(可有可無) date 抓出來 存成dict 在做後續的處理

        for dict_df in dict_list:
            if "data" not in dict_df and "title" not in dict_df and "fields" not in dict_df:
                continue
            elif not dict_df["data"] and dict_df["title"] != "" and dict_df["fields"]:
                continue
            elif not dict_df["data"] and dict_df["title"] is None and not dict_df["fields"]:
                continue

            # 先抓小分類，因為小分類攸關這個dict_df需不需要被執行
            dict_df["subitem"] = keyinstr(str=dict_df["title"], dic=colname_dic, lis=subtitle, default=dict_df["title"])

            # subtitle如果被改過，原本要下載的，已經不用下載了，就要用下面subtitle_new抓最新的subtitle做更改
            subtitle_new = [colname_dic.get(_,_) for _ in collection[file_info["parentdir"]]["subtitle"] ]
            if dict_df["subitem"] not in subtitle_new:
                continue

            dict_df["file_name"] = file
            dict_df["item"] = file.split("_")[0]
            dict_df["date"] = file_data["crawlerdic"]["payload"]["date"]
            # 用file_data的date會有錯20090113會變成00180113所以最安全是用crawlerdic裡面的
            dict_df["data_cleaned"] = pd.DataFrame()

            if dict_df["subitem"] in fields_span:
                data_cleaned_groups(dict_df)
            else:
                frameup_safe(dict_df)

            dict_df["data_cleaned"] = data_cleaned_df(dict_df["data_cleaned"],dict_df["item"],dict_df["subitem"],date=pd.to_datetime(dict_df["date"]))
            # break
            if "代號" not in dict_df["data_cleaned"]:
                tosql_df(df=dict_df["data_cleaned"], dbpath=join(dbpath_cleaned, dict_df["item"] + ".db"), table=dict_df["subitem"], pk=["代號","date"])
            else:
                tosql_df(df=dict_df["data_cleaned"], dbpath=join(dbpath_cleaned, dict_df["item"] + ".db"),
                         table=dict_df["subitem"], pk=[])
            # 放進db，用最簡單的模式，直覺型放入，沒有用adapter

        log.loc[file_data["crawlerdic"]["payload"]["date"] , file.split("_")[0]] = file
        picklesave(log, dbpath_cleaned_log)
        print("OK")
        #
        # break
        #
        # file_data = pickleload(r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source/發行量加權股價指數歷史資料/發行量加權股價指數歷史資料_2023-05-02.pkl")
        # file_data = pickleload(r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source/每月當日沖銷交易標的及統計/每月當日沖銷交易標的及統計_2020-09-30.pkl")
        # file_data = pickleload(r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source/每日收盤行情/每日收盤行情_2025-04-15.pkl")
        # file_data = pickleload(r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source/信用交易統計/信用交易統計_2023-04-14.pkl")
        #
        # test["data_cleaned"]={}
        # frameup_safe(test)
        # if "date" in test["data_cleaned"]["發行量加權股價指數歷史資料"]:
        #     test["data_cleaned"]["發行量加權股價指數歷史資料"].index = pd.to_datetime(test["data_cleaned"]["發行量加權股價指數歷史資料"]["日期"])
        #
        #
        # test1= pickleload(
        #     r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source/發行量加權股價指數歷史資料/發行量加權股價指數歷史資料_2023-05-01.pkl")
        # if not test["data"]:print("ok")
        #
        #
        #
        #
        # data = productdict(file_data, getkeys(file_data))
        # clean_manager = cleaner(data, file.split("_")[0])
        # file_data.keys()
        # file_data["data"]
        # file_data["title"]
        # print(file,path)


