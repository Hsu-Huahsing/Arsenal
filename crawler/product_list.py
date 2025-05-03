import requests as re
from bs4 import BeautifulSoup
import pandas as pd


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
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=1"
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