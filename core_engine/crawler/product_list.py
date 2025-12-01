# 重構後的 product_list.py 模組
"""
此模組負責從公開資訊觀測站網站抓取台灣證券市場的商品清單，將資料儲存於本地檔案，並提供函式介面供其他模組使用。
提供的功能包含：
- 從網路抓取各市場的商品清單資料並整理（loader）
- 載入本地已儲存的商品清單資料（loader）
- 更新本地商品清單資料（updater），將新抓取的資料合併至已有資料（具快取機制）
- 將整理後的商品清單資料儲存至檔案（saver）
- 提供 CLI 介面執行更新（執行 `python product_list.py --update`）
模組中的函式與類別均以中文註解詳細說明用途和參數。
範例使用:
    # 作為獨立執行 (CLI)
    # 終端機下執行，更新本地商品清單資料:
    python -m crawler.product_list --update
    # 或
    python crawler/product_list.py --update

    # 作為模組匯入使用
    from crawler import product_list
    product_list.main(['--update'])     # 直接更新
    # 或
    product_list.update_data()          # 呼叫函式跳過 argparse

    # 初始化/更新資料：第一次呼叫會抓取資料並建立本地檔案
    df = product_list.update_data()

    # 載入已存在的資料（不進行更新）
    df_existing = product_list.load_data()

    # 查詢資料：例如查詢代號為 '2330' 的股票
    result = df[df['代號'] == '2330']
    print(result)

    # 資料儲存格式：資料將以 pd.DataFrame 以 pickle 檔格式儲存在預設路徑 (參見 config.paths.dbpath_productlist)
    # 可透過 product_list.load_data() 載入，或使用 pandas 等工具讀取。
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from os import makedirs
from os.path import exists, dirname
from StevenTricks.control_flow import sleepteller
from StevenTricks.file_utils import pickleio
from config.conf import dailycollection, product_clean
from config.col_rename import colname_dic
from config.paths import dbpath_productlist
from StevenTricks.core.df_utils import DataFrameMerger
import argparse
from typing import Optional, List

def fetch_product_list(url: str) -> pd.DataFrame:
    """
    從指定的 URL 抓取商品清單表格，並回傳 pandas DataFrame。
    參數:
        url (str): 目標資料的完整網址。
    回傳:
        pandas.DataFrame: 包含抓取到的商品清單表格資料。
    例外:
        requests.exceptions.RequestException: 網路請求發生錯誤時拋出。
        ValueError: 當無法從回傳內容找到表格資料時拋出。
    """
    # 定義 HTTP 請求標頭，模擬一般使用者瀏覽器
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://isin.twse.com.tw/"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        # 網路請求錯誤（例如連線失敗、逾時等）
        raise requests.exceptions.RequestException(f"商品清單抓取失敗: {e}")
    # 確認 HTTP 回應狀態碼
    if response.status_code != 200:
        raise ValueError(f"無法取得商品清單，HTTP狀態碼: {response.status_code}")
    # 設定正確的回應編碼以確保中文顯示正常
    response.encoding = response.apparent_encoding
    # 解析 HTML 內容
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    if len(tables) < 2:
        # 預期網頁中至少有兩個表格，第二個表格包含商品清單資料
        raise ValueError("解析錯誤：找不到商品清單表格")
    data_table = tables[1]  # 取第二個表格（索引從0開始）
    # 取得表頭列的所有欄位名稱
    header_cells = data_table.find("tr").find_all("td")
    headers = [cell.get_text(strip=True) for cell in header_cells]
    # 取得資料列
    data_rows = []
    for tr in data_table.find_all("tr")[1:]:  # 跳過表頭列
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        # 確認資料列欄位數與表頭一致，避免不完整的資料
        if len(cols) == len(headers):
            data_rows.append(cols)
    # 將資料轉成 DataFrame 並回傳
    df = pd.DataFrame(data_rows, columns=headers)
    return df


def fetch_all_products() -> pd.DataFrame:
    """
    抓取所有市場的商品清單資料，經整理後合併為單一 DataFrame 回傳。
    此函式會依據設定的 dailycollection['stocklist'] 中的各市場參數，逐一抓取資料。
    回傳:
        pandas.DataFrame: 包含所有市場商品清單的合併資料表。
    例外:
        Exception: 當任一市場的資料抓取或處理失敗時拋出相關錯誤。
    """
    product_data = {}  # 用於暫存各市場的 DataFrame
    # 取得所有需抓取的市場代號清單和對應的 URL 模板
    try:
        model_list = dailycollection['stocklist']['modelis']
        url_template = dailycollection['stocklist']['url']
    except Exception as e:
        # dailycollection 未正確設置
        raise KeyError(f"無法取得 dailycollection 中 stocklist 的配置: {e}")
    # 逐一抓取每個市場的商品清單
    for model in model_list:
        print(f"開始抓取{model}")
        url = url_template.format(str(model))
        df = fetch_product_list(url)
        # 暫存原始 DataFrame
        product_data[model] = df
        # 每次抓取後稍作延遲，避免過於頻繁的請求
        sleepteller()
    # 整理各市場資料：統一欄位名稱並清理內容
    for model, df in product_data.items():
        # 欄位重新命名
        df = df.rename(columns=colname_dic)
        # 移除全形空格等不必要字元
        df.replace({"\u3000": ""}, regex=True, inplace=True)
        # 如果資料中有 ISINCode 欄位，根據 product_clean 的設定擷取股票代號
        if "ISINCode" in df.columns:
            try:
                code_range = product_clean["code"][model]
                df["代號"] = df["ISINCode"].str.slice(code_range[0], code_range[1])
            except KeyError:
                # 若 product_clean 缺少相應的設定，跳過擷取代號這一步
                pass
        # 將整理後的 DataFrame 賦回暫存字典
        product_data[model] = df
    # 將所有市場的資料合併為單一 DataFrame
    combined_df = pd.concat(list(product_data.values()), axis=0, ignore_index=True)
    return combined_df


def load_data() -> pd.DataFrame:
    """
    載入本地儲存的商品清單資料。
    回傳:
        pandas.DataFrame: 本地保存的商品清單 DataFrame。
    例外:
        FileNotFoundError: 找不到本地資料檔案時拋出。
        Exception: 如果資料無法正確載入或解析則拋出一般例外。
    """
    if not exists(dbpath_productlist):
        # 如果檔案不存在，提示使用者需要先更新資料
        raise FileNotFoundError(f"找不到本地商品清單資料檔案: {dbpath_productlist}。請先執行更新動作。")
    try:
        data = pickleio(path=dbpath_productlist, mode="load")
        # 確認資料載入結果是否為 DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("載入的資料格式不正確，非 DataFrame")
        return data
    except Exception as e:
        raise Exception(f"載入本地資料失敗: {e}")


def save_data(data: pd.DataFrame) -> None:
    """
    將商品清單 DataFrame 資料儲存至預定的本地路徑。
    如果目標資料夾不存在，將自動建立。
    參數:
        data (pd.DataFrame): 要儲存的商品清單資料。
    例外:
        Exception: 當資料儲存過程發生問題時拋出。
    """
    # 確保目標路徑的資料夾存在，若無則建立
    try:
        makedirs(dirname(dbpath_productlist), exist_ok=True)
    except Exception as e:
        raise Exception(f"建立資料夾失敗: {e}")
    # 使用 pickleio 進行資料儲存
    try:
        pickleio(data=data, path=dbpath_productlist, mode="save")
    except Exception as e:
        raise Exception(f"儲存資料失敗: {e}")


def update_data() -> pd.DataFrame:
    """
    抓取最新商品清單資料並更新本地儲存的資料。
    若本地尚無資料檔案，將自動建立新檔；若已有資料，則合併新資料以更新之。
    回傳:
        pandas.DataFrame: 更新後的完整商品清單 DataFrame。
    例外:
        Exception: 任一過程發生錯誤時拋出。
    """
    # 抓取最新的商品清單資料
    new_data = fetch_all_products()
    # 如果本地尚無舊資料，直接儲存新資料
    if not exists(dbpath_productlist):
        save_data(new_data)
        return new_data
    # 載入現有的舊資料
    try:
        old_data = pickleio(path=dbpath_productlist, mode="load")
        if not isinstance(old_data, pd.DataFrame):
            # 如果舊資料格式不正確，改為直接使用新資料覆蓋
            old_data = pd.DataFrame()
    except Exception as e:
        # 無法讀取舊資料，記錄錯誤但繼續使用空的 DataFrame
        print(f"載入舊資料時發生錯誤，將使用新資料重新建立: {e}")
        old_data = pd.DataFrame()
    # 使用 DataFrameMerger 將新資料合併入舊資料
    merger = DataFrameMerger(old_data)
    updated_data = merger.renew(new_data, overwrite=True)
    # 將更新後的資料儲存回檔案
    save_data(updated_data)
    return updated_data




def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="商品清單抓取與更新工具")
    parser.add_argument('--update', action='store_true', help='抓取最新商品清單並更新本地資料')
    # 只解析你給的 argv；若沒給，就當成空清單；忽略未知參數
    args, _unknown = parser.parse_known_args(argv or [])

    if args.update:
        print("開始更新商品清單資料...")
        try:
            updated_df = update_data()
            print(f"商品清單更新完成，總計 {len(updated_df)} 筆記錄已儲存至 {dbpath_productlist}")
        except Exception as e:
            print(f"更新過程發生錯誤: {e}")
    else:
        print("請使用 --update 參數來更新商品清單資料。例如:")
        print("  python product_list.py --update")

# 2) 只有當「直接以腳本執行」時，才吃真正的命令列參數
if __name__ == "__main__":
    main(['--update'])


    # test=pickleio(dbpath_productlist, mode="load")