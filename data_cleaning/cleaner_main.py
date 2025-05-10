import pandas as pd
from data_cleaning.fundic_mapping import fundic
from StevenTricks.convert_utils import findbylist
from conf import collection,dbpath,dbpath_productlist,dbpath_log,dbpath_source,dbpath_cleaned
from StevenTricks.file_utils import logfromfolder,  picklesave, pickleload, sweep_path, PathWalk_df
from os.path import join
from schema_utils import productdict, getkeys, safe_frameup_data

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
        file_data = pickleload(path)
        file_info = sweep_path(path)
        clean_type = fundic[file_info["parentdir"]][file.split("_")[0]]
        # 因為有時候裡面的欄位(fields)數量和實際的data長度不一樣，會出錯，所以要用safe的方式去做出dataframe，自動先做出一樣長度的架構，再把多餘長度的欄位刪掉，未來如果發現有誤刪，就要再修正
        data_raw = safe_frameup_data(file_data["data"], file_data["fields"])

        data_cleaned = clean_type(data_raw,file_info["parentdir"],file.split("_")[0],file_name=file)
        break

        data = productdict(file_data, getkeys(file_data))
        clean_manager = cleaner(data, file.split("_")[0])
        file_data.keys()
        file_data["data"]
        file_data["title"]
        print(file,path)


