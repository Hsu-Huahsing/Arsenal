import pandas as pd
from data_cleaning.cleaning_utils import data_cleaned_df, data_cleaned_groups, frameup_safe,dict_extract
from StevenTricks.convert_utils import findbylist
from StevenTricks.dbsqlite import tosql_df
from conf import collection,dbpath_source,dbpath_cleaned,dbpath_cleaned_log
from StevenTricks.file_utils import picklesave, pickleload, sweep_path, PathWalk_df
from os.path import join
from itertools import chain


def cleaner(product, title):
    """
    根據 title 取出每個 subtitle 對應的清洗函式，處理 product 資料
    """
    result = {}
    for key, df in product.items():
        mapped = findbylist(collection[title]['subtitle'], key)
        if not mapped:
            print(f"[Warning] {key} is not in collection[{title}]['subtitle']")
            continue
        if len(mapped) > 1:
            print(f"[Warning] {key} maps to multiple subtitles: {mapped}")
            continue
        subtitle = mapped[0]
        try:
            func = fundic[title][subtitle]
            cleaned = func(df, title, subtitle)
            result.update(cleaned)
        except Exception as e:
            print(f"[Error] Failed cleaning {title} - {subtitle} with error: {e}")
            continue
    return result


log_info = sweep_path(dbpath_cleaned_log)

if __name__ == "__main__":
    dbpath_list = PathWalk_df(dbpath_source, [], ["log"], [], [".pkl"])
    if log_info["exists"] is True:
        log = pickleload(dbpath_cleaned_log)
        log = log.values.tolist()
        log = list(chain.from_iterable(log))
        dbpath_list = dbpath_list.loc[~dbpath_list["file"].isin(log),:]
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

        dict_list = []

        if "tables" in file_data:
            for table in file_data["tables"]:
                if table:
                    dict_list.append(dict_extract(table,date=file_data["date"]))
        else:
            dict_list.append(dict_extract(file_data, date=file_data["date"]))

        if "creditTitle" in file_data :
            if file_data["creditTitle"] is not None:
                dict_list.append(dict_extract(file_data, title="creditTitle", fields="creditFields", data="creditList",  date=file_data["date"]))

        if not dict_list:
            continue
        # 用抓table的方式，把固定的格式 title fields data groups(可有可無) date 抓出來 存成dict 在做後續的處理

        for dict_df in dict_list:
            dict_df["file_name"] = file
            dict_df["item"] = file.split("_")[0]
            # dict_df["subitem"] = file.split("_")[0]
            dict_df["data_cleaned"] = pd.DataFrame()
            frameup_safe(dict_df)

            if "groups" in dict_df:
                dict_df = data_cleaned_groups(dict_df)

            dict_df["data_cleaned"] = data_cleaned_df(dict_df["data_cleaned"],dict_df["item"],dict_df["subitem"],date=pd.to_datetime(dict_df["date"]))

            tosql_df(df=dict_df["data_cleaned"], dbpath=join(dbpath_cleaned, dict_df["item"] + ".db"), table=dict_df["subitem"], pk=[])
            # 放進db，用最簡單的模式，直覺型放入，沒有用adapter

        log.loc[file_data["date"] , file.split("_")[0]] = file
        #
        # break
        #
        # file_data = pickleload(r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source/發行量加權股價指數歷史資料/發行量加權股價指數歷史資料_2023-05-02.pkl")
        #
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


