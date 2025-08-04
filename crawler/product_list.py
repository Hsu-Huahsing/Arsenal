import requests as re
from bs4 import BeautifulSoup
import pandas as pd
from StevenTricks.control_flow import sleepteller
from os.path import exists
from StevenTricks.file_utils import pickleio
from config.conf import dailycollection, product_clean
from config.col_rename import colname_dic
from config.paths import dbpath_productlist
from StevenTricks.df_utils import DataFrameMerger

def product_list(url):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://isin.twse.com.tw/"
    }
    res = re.get(url, headers=headers)
    res.encoding = res.apparent_encoding
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find_all("table")
    rows = []
    data_table = table[1]
    header_row = data_table.find("tr")
    headers = [td.get_text(strip=True) for td in header_row.find_all("td")]

    # 擷取表格資料
    rows = []
    for tr in data_table.find_all("tr")[1:]:  # 跳過表頭
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) == len(headers):  # 確保列長度一致
            rows.append(cols)

    # 建立 DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df


if __name__ == "__main__":
    # url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=1"
    # headers = {
    #     "User-Agent": "Mozilla/5.0",
    #     "Referer": "https://isin.twse.com.tw/"
    # }
    # res = re.get(url, headers=headers)
    # res.encoding = res.apparent_encoding
    # soup = BeautifulSoup(res.text, "html.parser")
    # table = soup.find_all("table")
    # rows = []
    # data_table = table[1]
    # header_row = data_table.find("tr")
    # headers = [td.get_text(strip=True) for td in header_row.find_all("td")]
    #
    # # 擷取表格資料
    # rows = []
    # for tr in data_table.find_all("tr")[1:]:  # 跳過表頭
    #     cols = [td.get_text(strip=True) for td in tr.find_all("td")]
    #     if len(cols) == len(headers):  # 確保列長度一致
    #         rows.append(cols)
    #
    # # 建立 DataFrame
    # df = pd.DataFrame(rows, columns=headers)

    product_dic = {}
    # 先進行商品清單下載
    for _ in dailycollection['stocklist']['modelis']:
        product = product_list(dailycollection['stocklist']['url'].format(str(_)))
        product_dic[_] = product
        print(_)
        sleepteller()
    # 針對商品清單做資料清理
    for key in product_dic:
        product_df = product_dic[key]
        product_df = product_df.rename(columns=colname_dic)
        product_df = product_df.replace({"\u3000": ""}, regex=True)
        if "ISINCode" in product_df:
            product_df["代號"] = product_df["ISINCode"].str.slice(product_clean["code"][key][0],
                                                                  product_clean["code"][key][1])
        product_dic[key] = product_df

    product = pd.concat(list(product_dic.values()), axis=0)
    if exists(dbpath_productlist) is False:
        pickleio(data=product, path=dbpath_productlist, mode="save")
    else:
        product_old = pickleio(dbpath_productlist, mode="load")
        product_manager = DataFrameMerger(product_old)
        product_old = product_manager.renew(product)
        pickleio(data=product_old, path=dbpath_productlist, mode="save")