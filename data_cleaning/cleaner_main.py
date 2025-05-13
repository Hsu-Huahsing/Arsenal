import pandas as pd
from cytoolz import remove

from data_cleaning.fundic_mapping import fundic
from data_cleaning.cleaning_utils import data_cleaned_pre, data_cleaned_df, data_cleaned_groups, frameup_safe,dict_extract
from StevenTricks.convert_utils import findbylist,changetype_stringtodate
from conf import collection,dbpath,dbpath_productlist,dbpath_log,dbpath_source,dbpath_cleaned,datecol
from StevenTricks.file_utils import logfromfolder,  picklesave, pickleload, sweep_path, PathWalk_df
from os.path import join
from pathlib import Path

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

if __name__ == "__main__":
    dbpath_list = PathWalk_df(dbpath_source, [], ["log"], [], [".pkl"])
    for file , path in dbpath_list[["file","path"]].values:
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

        if "creditTitle" in file_data:
            dict_list.append(dict_extract(file_data, title="creditTitle", fields="creditFields", data="creditList",  date=file_data["date"]))

        if not dict_list:
            continue
        # 用抓table的方式，把固定的格式 title fields data groups(可有可無) date 抓出來 存成dict 在做後續的處理

        for dict_df in dict_list:



            file_data["file_name"] = file
            file_data["item"] = file_info["parentdir"]
            file_data["subitem"] = file.split("_")[0]
            file_data["data_cleaned"] = {}
            data_cleaned_pre(file_data)

        frameup_safe(file_data)

        if "creditList" in file_data:
            file_data["data_cleaned"]["creditTitle"] = pd.DataFrame(file_data["creditList"], columns=file_data["creditFields"])

        if "groups" in file_data:
            file_data = data_cleaned_groups(file_data)

        for key in file_data["data_cleaned"]:
            file_data["data_cleaned"][key] = data_cleaned_df(file_data["data_cleaned"][key],file_data["item"],file_data["subitem"],date=pd.to_datetime(file_data["date"]))


        break

        test = pickleload(r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source/發行量加權股價指數歷史資料/發行量加權股價指數歷史資料_2023-05-02.pkl")


        test["data_cleaned"]={}
        frameup_safe(test)
        if "date" in test["data_cleaned"]["發行量加權股價指數歷史資料"]:
            test["data_cleaned"]["發行量加權股價指數歷史資料"].index = pd.to_datetime(test["data_cleaned"]["發行量加權股價指數歷史資料"]["日期"])


        test1= pickleload(
            r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse/source/發行量加權股價指數歷史資料/發行量加權股價指數歷史資料_2023-05-01.pkl")
        if not test["data"]:print("ok")




        data = productdict(file_data, getkeys(file_data))
        clean_manager = cleaner(data, file.split("_")[0])
        file_data.keys()
        file_data["data"]
        file_data["title"]
        print(file,path)


