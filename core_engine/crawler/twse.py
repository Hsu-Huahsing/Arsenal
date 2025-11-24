#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
此模組整合了 crawler_main.py 與 stock.py 的功能，用於抓取台灣證券交易所（TWSE）的歷史股價及相關統計資料。
提供以下功能：
- 自動更新所有設定項目的歷史資料（包含每日收盤行情、信用交易統計等，參見 config.conf 中的 `collection` 定義），並將資料儲存至本地檔案。
- 抓取單一股票的最近 N 日歷史股價資料（例如提供股票代號與天數），以 DataFrame 格式回傳，方便獨立測試與除錯。
- 記錄抓取狀態至日誌檔（log.pkl）以及錯誤紀錄檔（errorlog.pkl），每次執行自動檢查/更新這些日誌，以避免重複抓取已取得的資料。

範例使用:
    # 1. 作為獨立腳本執行 (CLI)
    # 更新所有台股歷史資料（將自動檢查並抓取缺少的部分）:
    python -m crawler.twse --update
    # 或僅抓取特定股票近 90 日的資料:
    python -m crawler.twse --symbol 2330 --days 90

    # 2. 作為模組匯入使用
    from crawler import twse
    # 抓取單一股票資料範例 (2330 台積電 最近90日):
    df = twse.fetch_stock('2330', days=90)
    print(df.head())  # 顯示前幾筆資料
    # 更新所有資料:
    twse.update_data()  # 執行更新流程，等價於 CLI 模式的 --update
"""

import sys
import datetime
import logging
import argparse
from typing import Optional, List
import pandas as pd
from os import makedirs, remove
from os.path import join, exists
from traceback import format_exc

# 匯入自定義工具和設定
from StevenTricks.df_utils import periodictable, findval
from StevenTricks.file_utils import logfromfolder, pickleio, PathWalk_df
from StevenTricks.net_utils import randomheader
from StevenTricks.control_flow import sleepteller
from config.conf import collection
from config.paths import db_root, dbpath_source, dbpath_log, dbpath_errorlog
from schema_utils import warehouseinit

# 設定全域 logger
logger = logging.getLogger(__name__)
# 預設把 root logger 設成 DEBUG（若外部尚未設定 logging）
_root = logging.getLogger()
if not _root.handlers:  # 避免重複加 handler
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger.setLevel(logging.DEBUG)  # 本模組 logger 也設為 DEBUG

class CrawlerConfig:
    """
    封裝爬蟲任務需要的所有變數與資源。
    """
    def __init__(self, collection: dict, log: pd.DataFrame, errorlog: pd.DataFrame,
                 dbpath_source: str, dbpath_log: str, dbpath_errorlog: str):
        self.collection = collection            # 爬蟲任務配置（各項資料的請求設定）
        self.log = log                          # 任務執行狀態記錄表 (DataFrame)
        self.errorlog = errorlog                # 錯誤紀錄表 (DataFrame)
        self.dbpath_source = dbpath_source      # 原始資料儲存目錄
        self.dbpath_log = dbpath_log            # log 檔案路徑
        self.dbpath_errorlog = dbpath_errorlog  # error log 檔案路徑
        # 外部模組/函式
        self.randomheader = randomheader        # 隨機 header 生成函式
        self.sleepteller = sleepteller          # 控制爬蟲休眠的函式

        # 使用 requests 模組，但為方便除錯，可透過此屬性替換為測試用的 request 函式
        import requests
        self.request_module = requests

class CrawlerTask:
    """
    執行單筆爬蟲任務，處理資料請求、儲存，以及 log/errorlog 紀錄。
    """
    def __init__(self, config: CrawlerConfig):
        self.cfg = config

    def run_task(self, ind, col):
        """
        執行指定日期 (ind) 和資料類型 (col) 的爬蟲任務。
        將請求結果儲存至檔案，並即時更新 log/errorlog 狀態。
        參數:
            ind: 日期索引（字串格式 YYYY-MM-DD），對應要抓取的日期。
            col: 資料項目名稱（對應 collection 的鍵值）。
        """
        # 準備該資料項目的設定並設定日期參數
        # 新增：標準化日期鍵
        key = pd.to_datetime(ind).strftime("%Y-%m-%d")

        crawlerdic = self.cfg.collection[col].copy()  # 取得爬蟲設定字典 (複製一份避免污染全域設定)

        crawlerdic['payload']['date'] = key.replace('-', '')  # 改用 key  # API 要求的日期格式為 YYYYMMDD

        # 建立該類別資料的儲存目錄（若不存在）
        datapath = join(self.cfg.dbpath_source, col)
        makedirs(datapath, exist_ok=True)
        logger.info(f"抓取 {key} 的「{col}」資料...")

        try:
            # 發送 HTTP POST 請求取得資料
            res = self.cfg.request_module.post(
                url=crawlerdic['url'],
                headers=next(self.cfg.randomheader()),
                data=crawlerdic['payload'],
                timeout=(3, 7)
            )
            data = res.json()  # 嘗試將回應內容轉為 JSON
            logger.debug("已收到回應，暫停一段時間以避免觸發防爬蟲機制...")
            self.cfg.sleepteller()  # 短暫休眠

        except KeyboardInterrupt:
            # 使用者手動中斷，保存目前 log 狀態後退出
            logger.warning("偵測到手動中斷，正在儲存目前進度...")
            pickleio(data=self.cfg.log, path=self.cfg.dbpath_log, mode="save")
            pickleio(data=self.cfg.errorlog, path=self.cfg.dbpath_errorlog, mode="save")
            logger.info("已儲存 log 至硬碟，程序中止。")
            sys.exit(1)

        except Exception as e:
            # 諸如連線失敗、JSON 解碼錯誤或被反爬截斷等未知錯誤
            logger.error(f"資料抓取過程發生錯誤: {e}")
            logger.debug(format_exc())  # 輸出詳細的堆疊追蹤於 debug 日誌
            # 在 log 標記錯誤，在 errorlog 記錄詳細資訊
            self.cfg.log.loc[self.cfg.log.index == key, col] = 'request error'
            errordic = {
                'crawlerdic': crawlerdic,
                'errormessage1': format_exc(),
                'errormessage2': str(e),
                'errormessage3': 'request failed'
            }
            # 將錯誤細節以物件形式存入 errorlog 的該日期欄位 (使用 list 包裝以存入 DataFrame)
            self.cfg.errorlog.loc[key, col] = [errordic]
            # 儲存 log/errorlog 狀態
            pickleio(data=self.cfg.log, path=self.cfg.dbpath_log, mode="save")
            pickleio(data=self.cfg.errorlog, path=self.cfg.dbpath_errorlog, mode="save")
            # 發生錯誤時適當延長休眠時間再繼續，以免持續碰撞相同錯誤
            self.cfg.sleepteller(mode='long')
            return  # 此筆任務終止，但迴圈可繼續下一筆

        # 處理 HTTP 回應代碼
        if res.status_code == self.cfg.request_module.codes.ok:
            # 只有在 200 OK 的情況下才繼續處理資料
            if data.get('stat') == 'OK':
                # 資料正常，標記該日期該項目成功
                self.cfg.log.loc[self.cfg.log.index == key, col] = 'succeed'
            else:
                # 資料取得不完整，例如該日期為非交易日，標記為 'close'
                self.cfg.log.loc[self.cfg.log.index == key, col] = 'close'
                pickleio(data=self.cfg.log, path=self.cfg.dbpath_log, mode="save")
                logger.info(f"{key} 非交易日或無資料，略過存檔。")
                return  # 不保存任何資料檔案

        else:
            # 非 200 的 HTTP 回應（如 403 禁止訪問或 500 伺服器錯誤）
            self._handle_status_error(key, col, crawlerdic, res)
            # 拋出例外中止剩餘任務（對於被禁止訪問等情況，中止整個更新流程）
            raise Exception(f"HTTP {res.status_code} 錯誤，終止爬蟲任務")

        # 如果程式走到這，表示成功取得有效的資料，將資料存檔
        data['crawlerdic'] = crawlerdic  # 保存當時使用的設定參數，方便日後調試
        data['request'] = res            # 保存 request 物件以供需要時查閱
        filename = join(datapath, f"{col}_{key}.pkl")
        pickleio(data=data, path=filename, mode="save")
        logger.debug(f"資料已儲存至 {filename}")

        # 若此項目的資料頻率為月頻，只保留本日資料，刪除本月較早的資料檔案
        if crawlerdic.get('freq') == 'M':
            self._clean_monthly_files(key, col, datapath)

        # 每完成一筆資料抓取，立即儲存更新後的 log（確保狀態持久化）
        pickleio(data=self.cfg.log, path=self.cfg.dbpath_log, mode="save")

    def _handle_status_error(self, ind, col, crawlerdic, res):
        """
        處理 HTTP 狀態錯誤（如 403 或 500）。記錄錯誤資訊到 log 和 errorlog。
        """
        logger.error(f"HTTP 狀態碼錯誤: {res.status_code} (日期: {ind}, 項目: {col})")
        # 在 log 中標記
        self.cfg.log.loc[self.cfg.log.index == ind, col] = 'result error'
        # 在 errorlog 中記錄詳情
        errordic = {
            'crawlerdic': crawlerdic,
            'request': res,
            'requeststatus': res.status_code,
            'errormessage1': 'result error'
        }
        self.cfg.errorlog.loc[ind, col] = [errordic]
        # 儲存最新的 log 和 errorlog
        pickleio(data=self.cfg.log, path=self.cfg.dbpath_log, mode="save")
        pickleio(data=self.cfg.errorlog, path=self.cfg.dbpath_errorlog, mode="save")
        # 不在此函式終止程式，由呼叫者決定（透過拋出例外）

    @staticmethod
    def _clean_monthly_files( ind, col, datapath):
        """
        針對月頻資料，只保留當月最新資料檔案，刪除該月較舊的檔案。
        """
        # 產生從當月月初到 ind 前一天的日期範圍
        start_date = ind[:8] + "01"  # 該月份第一天 (YYYY-MM-01)
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(ind, "%Y-%m-%d").date() - datetime.timedelta(days=1)
        daterange = pd.date_range(start=start_date, end=end_date, freq='D')
        for d in daterange:
            oldfile = join(datapath, f"{col}_{d.strftime('%Y-%m-%d')}.pkl")
            if exists(oldfile):
                try:
                    remove(oldfile)
                    logger.debug(f"已刪除過期檔案: {oldfile}")
                except Exception as e:
                    logger.warning(f"無法刪除檔案 {oldfile}: {e}")

def fetch_stock(symbol: str, days: int) -> pd.DataFrame:
    """
    抓取指定股票在最近 days 天內的每日股價資料，回傳 pandas DataFrame。
    參數:
        symbol (str): 股票代號 (如 "2330")，僅限臺股上市股票代號。
        days (int): 要抓取的天數範圍（向過去數的天數，包含當天）。
    回傳:
        pandas.DataFrame: 包含日期、成交股數、成交金額、開盤價、最高價、最低價、收盤價、漲跌價差、成交筆數等欄位的資料表。
        若指定區間無交易資料（如股票尚未上市或非交易日），則可能回傳空的 DataFrame。
    例外:
        requests.exceptions.RequestException: 網路請求失敗時拋出，例如連線錯誤或逾時。
        ValueError: 輸入參數不合規定（例如 days 非正數）時拋出。
    使用範例:
        df = fetch_stock("2330", days=30)
        print(df.tail())  # 顯示抓取的最後幾天資料
    """
    if days <= 0:
        raise ValueError("days 必須為正整數")

    # 計算起始及結束日期
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days-1)
    # 生成包含起始到結束月份的列表，用於逐月抓取
    months = []
    cur = datetime.date(start_date.year, start_date.month, 1)
    end_month = datetime.date(end_date.year, end_date.month, 1)
    while cur <= end_month:
        months.append(cur)
        # 下個月的第一天
        year = cur.year + (cur.month // 12)
        month = cur.month % 12 + 1
        cur = datetime.date(year, month, 1)

    # 準備暫存結果的列表
    data_frames = []
    import requests
    base_url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json"
    for month_start in months:
        date_param = month_start.strftime("%Y%m01")  # 該月的第一天 (YYYYMM01)
        url = f"{base_url}&date={date_param}&stockNo={symbol}"
        logger.info(f"向 TWSE 抓取 {symbol} 股票 {month_start.year}年{month_start.month}月 的資料...")
        try:
            resp = requests.get(url, timeout=10)
        except requests.exceptions.RequestException as e:
            logger.error(f"網路請求失敗: {e}")
            raise  # 將網路相關錯誤拋出，由上層處理
        if resp.status_code != 200:
            logger.error(f"HTTP 狀態碼 {resp.status_code} 錯誤，無法取得 {symbol} 的資料")
            raise ValueError(f"無法取得 {symbol} 的資料（HTTP {resp.status_code}）")
        resp.encoding = resp.apparent_encoding  # 確保中文編碼正確
        data_json = resp.json()
        if not data_json.get('data'):
            logger.warning(f"{month_start.year}-{month_start.month:02d} 沒有 {symbol} 的交易資料")
            continue  # 該月無資料（可能該股票尚未上市或該月全為非交易日），跳過
        # 將當月資料轉為 DataFrame
        fields = data_json.get('fields') or [
            "日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價",
            "收盤價", "漲跌(+/-)", "漲跌價差", "成交筆數"
        ]
        df_month = pd.DataFrame(data_json['data'], columns=fields)
        data_frames.append(df_month)
        # 每抓取一個月暫停幾秒，避免過於頻繁（使用短暫 sleep）
        sleepteller()

    if not data_frames:
        # 若沒有任何資料，回傳空 DataFrame
        return pd.DataFrame()

    # 合併所有月份資料並篩選目標日期範圍
    df_all = pd.concat(data_frames, ignore_index=True)
    # 將日期由民國年轉為西元年並格式化為 YYYY-MM-DD 字串
    def convert_date(roc_date: str) -> str:
        # 民國年日期格式: "yyy/mm/dd" (例如 "110/08/05")
        parts = roc_date.strip().split('/')
        if len(parts) != 3:
            return roc_date  # 無法辨識的格式，維持原值
        year = int(parts[0]) + 1911
        month = int(parts[1])
        day = int(parts[2])
        return f"{year}-{month:02d}-{day:02d}"
    df_all['日期'] = df_all['日期'].apply(convert_date)
    # 篩選指定的日期區間
    mask = (df_all['日期'] >= start_date.strftime("%Y-%m-%d")) & (df_all['日期'] <= end_date.strftime("%Y-%m-%d"))
    df_all = df_all.loc[mask].reset_index(drop=True)
    # 清理數據：去除逗號，轉換數值欄位型態
    numeric_cols_int = ["成交股數", "成交金額", "成交筆數"]
    numeric_cols_float = ["開盤價", "最高價", "最低價", "收盤價", "漲跌價差"]
    for col in numeric_cols_int:
        if col in df_all.columns:
            df_all[col] = df_all[col].str.replace(',', '')
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce').astype('Int64')
    for col in numeric_cols_float:
        if col in df_all.columns:
            df_all[col] = df_all[col].str.replace(',', '')
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
    # 處理漲跌符號欄位：將符號合併到漲跌價差數值中，然後移除符號欄
    sign_col = "漲跌(+/-)"
    change_col = "漲跌價差"
    if sign_col in df_all.columns and change_col in df_all.columns:
        # 用 '-' 號判斷跌或漲 (非 '-' 一律視為正號)
        df_all[change_col] = df_all.apply(
            lambda row: -row[change_col] if str(row[sign_col]).strip() == '-' else row[change_col],
            axis=1
        )
        df_all.drop(columns=[sign_col], inplace=True)
    # 重新調整欄位順序 (若移除了符號欄，確保其餘欄位順序合理)
    # 將 '日期' 設為第一欄
    cols = list(df_all.columns)
    if '日期' in cols:
        cols.insert(0, cols.pop(cols.index('日期')))
        df_all = df_all[cols]
    return df_all

def update_data() -> None:
    """
    更新所有 TWSE 資料集的歷史資料。
    此函式會：
    1. 初始化資料目錄（如不存在則建立）。
    2. 載入或建立 log.pkl（抓取進度記錄表）和 errorlog.pkl（錯誤記錄表）。
    3. 更新 log 表：標記已有的資料，找出尚未抓取或需更新的日期。
    4. 逐筆抓取尚未完成的資料，過程中即時更新並儲存 log/errorlog。
    執行完畢後，所有 collection 定義的資料都會更新至最新日期，log.pkl/errorlog.pkl 也會更新存盤。
    可能拋出的例外將由呼叫者處理（例如網路錯誤時會拋出異常）。
    """
    logger.info("初始化資料夾及日誌檔...")
    warehouseinit(db_root)  # 確保倉庫資料夾存在
    # 載入或初始化 log DataFrame
    if exists(dbpath_log):
        log = pickleio(path=dbpath_log, mode="load")

        # 轉換型別，正常都要能轉換，有報錯就寫額外資料清理，這裡故意讓他報錯，才會知道有哪些例外狀況，這裡是留歷史資料，不會有錯，當初在下載和寫入的時候都已經統一規格，所以這裡不應該出錯，出錯要人工查看
        log.index = pd.to_datetime(log.index, errors='coerce').strftime("%Y-%m-%d")

        # 若 log 沒有今天的紀錄，使用 periodictable 產生從最後日期+1 到今日的缺漏記錄
        last_date_str = max(pd.to_datetime(log.index, errors='coerce'))
        if pd.isna(last_date_str):
            last_date_str = datetime.date.today() - datetime.timedelta(days=1)
        else:
            last_date_str = last_date_str.date()
        today_str = datetime.date.today().isoformat()
        if today_str not in log.index:
            logger.info(f"log 缺少 {today_str} 的紀錄，正在更新 log 表...")
            missing_log = periodictable(collection, datemin=last_date_str + datetime.timedelta(days=1))
            missing_log.index = missing_log.index.strftime("%Y-%m-%d")
            # 將新產生的部分合併進 log DataFrame
            log = pd.concat([log, missing_log])
        logger.info("現有 log.pkl 載入成功")
    else:
        # 沒有 log 檔則重新建立整份 log 表
        log = periodictable(collection)
        log.index = log.index.strftime("%Y-%m-%d")  # 格式化索引為字串日期
        logger.info("log.pkl 不存在，已重新建立 log 表")

    # 載入或初始化 errorlog DataFrame
    if exists(dbpath_errorlog):
        errorlog = pickleio(path=dbpath_errorlog, mode="load")
    else:
        errorlog = pd.DataFrame()

    # 每次爬蟲前，更新 log 狀態以符合目前硬碟上已有的資料檔案
    logger.info("開始更新 log 檔案狀態...")
    # 列出資料夾中所有現有的資料檔（排除 log 相關檔案與隱藏檔）
    dbpath_list = PathWalk_df(dbpath_source, [], ["log"], [".DS"], [])
    # 將已有的檔案對應的日期在 log 中標記為 succeed
    log = logfromfolder(dbpath_list, log=log, fillval='succeed')
    logger.info("log 檔案更新完成，開始執行網路資料抓取...")

    # 執行爬蟲任務主迴圈
    config = CrawlerConfig(collection=collection, log=log, errorlog=errorlog,
                            dbpath_source=str(dbpath_source), dbpath_log=str(dbpath_log),
                            dbpath_errorlog=str(dbpath_errorlog))
    crawler = CrawlerTask(config)
    tasks = list(findval(log, 'wait'))  # 找出所有狀態為 'wait' 的 (日期, 項目)
    total_tasks = len(tasks)
    if total_tasks == 0:
        logger.info("目前沒有需要更新的資料，一切都是最新狀態。")
    else:
        logger.info(f"共有 {total_tasks} 筆資料需要抓取更新。")
    for ind, col in tasks:
        try:
            crawler.run_task(ind, col)
        except Exception as e:
            # 紀錄發生錯誤的任務索引與類型
            logger.error(f"任務過程中止於 {ind} - {col}: {e}")
            # 檢查是否為日期轉換的特殊錯誤
            if isinstance(e, ValueError) and "strftime" in str(e):
                # 1. 原始 ind 值與型別
                logger.error(f"原始 ind 值: {repr(ind)} (類型: {type(ind).__name__})")
                # 2. 錯誤任務的資料項目名稱
                logger.error(f"該任務的資料項目 (col): {col}")
                # 3. 完整的 traceback
                logger.error("完整的 traceback:\n" + format_exc())
                # 4. log DataFrame 中該索引的列內容
                if pd.isna(ind):
                    # ind 為 NaN 類型，需特別處理
                    problem_rows = config.log[config.log.index.isna()]
                    if not problem_rows.empty:
                        logger.error(f"log 資料表中索引為 NaN 的列資料:\n{problem_rows}")
                    else:
                        logger.error("log 資料表中沒有索引為 NaN 的資料列。")
                else:
                    try:
                        row_data = config.log.loc[[ind]]
                        logger.error(f"log 資料表中索引 {ind} 的列資料:\n{row_data}")
                    except Exception as e2:
                        logger.error(f"無法取得 log 中索引 {ind} 的列資料: {e2}")
                logger.error("因日期轉換錯誤，程式已停止執行。")
                # **不**儲存 log.pkl 和 errorlog.pkl，直接Raise中止
            else:
                # 非日期轉換類錯誤，維持原行為：保存日誌檔案後再中止
                pickleio(data=config.log, path=config.dbpath_log, mode="save")
                pickleio(data=config.errorlog, path=config.dbpath_errorlog, mode="save")
            raise  # 中止迴圈，向上拋出異常

    # 爬蟲任務結束後，統一將 log 和 errorlog 儲存
    pickleio(data=config.log, path=config.dbpath_log, mode="save")
    pickleio(data=config.errorlog, path=config.dbpath_errorlog, mode="save")
    logger.info(f"更新完成！最新 log 已儲存至 {dbpath_log}")

def main(argv: Optional[List[str]] = None) -> None:
    """
    解析命令列參數並執行相應動作的主函式。
    可接受的參數:
      --update           執行更新所有資料的操作
      --symbol <代號>    指定要抓取的單一股票代號（需配合 --days）
      --days <天數>      指定要抓取單一股票的天數範圍（需配合 --symbol）
    若未提供任何參數，將提示使用方法。
    """
    parser = argparse.ArgumentParser(description="TWSE歷史資料抓取與更新工具")
    parser.add_argument('--update', action='store_true', help='更新所有 TWSE 定義資料至最新')
    parser.add_argument('--symbol', type=str, help='指定股票代號，例如 2330')
    parser.add_argument('--days', type=int, help='指定抓取天數，例如 90')
    args, _ = parser.parse_known_args(argv)

    # 設定 logging 級別和格式
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers: h.setLevel(logging.DEBUG)  # 已存在的 handler 一併拉高

    global logger
    logger = logging.getLogger(__name__)

    # 參數檢查與對應行為
    if args.update:
        if args.symbol or args.days:
            parser.error("--update 不能與 --symbol/--days 同時使用")
        logger.info("開始更新所有 TWSE 資料...")
        try:
            update_data()
            logger.info(f"所有資料更新完成，記錄已更新至 {dbpath_log}")
        except Exception as e:
            logger.exception(f"更新過程發生錯誤: {e}")
    elif args.symbol or args.days:
        # 必須同時提供 symbol 和 days
        if not (args.symbol and args.days):
            parser.error("請同時提供 --symbol <代號> 和 --days <天數>")
        symbol = str(args.symbol)
        days = args.days
        logger.info(f"開始抓取股票 {symbol} 過去 {days} 天的資料...")
        try:
            df = fetch_stock(symbol, days)
            if df.empty:
                logger.warning(f"找不到 {symbol} 在最近 {days} 天內的交易資料。")
            else:
                logger.info(f"{symbol} 最近 {days} 天共有 {len(df)} 筆交易日資料:")
                print(df.to_string(index=False))
        except Exception as e:
            logger.exception(f"抓取股票資料時發生錯誤: {e}")
    else:
        # 未提供任何有效參數，列出使用說明
        print("請使用以下參數來執行相應功能。例如:")
        print("  python twse.py --update           # 更新所有台股歷史資料")
        print("  python twse.py --symbol 2330 --days 90   # 抓取指定股票近90日資料")
        parser.print_help()

# 當直接執行此腳本時，呼叫 main() 處理命令列參數。
if __name__ == "__main__":
    main(['--update'])
