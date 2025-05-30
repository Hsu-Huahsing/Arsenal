# === 基礎套件與自定義工具 ===
import sys
import datetime
import pandas as pd
from os import makedirs, remove
from os.path import join, exists
from traceback import format_exc
import requests

# 匯入你自己的工具套件（StevenTricks）
from StevenTricks.dfi import findval                 # 找出 log 中需要處理的資料
from StevenTricks.netGEN import randomheader         # 隨機 HTTP 請求 header（模擬瀏覽器）
from StevenTricks.file_utils import picklesave       # 儲存 .pkl 檔案工具
from StevenTricks.process import sleepteller         # 控制 sleep 時間，避免觸發反爬蟲

# === 配置類別 ===
class CrawlerConfig:
    """
    封裝爬蟲任務需要的所有變數與資源。
    """
    def __init__(self, collection, log, errorlog,
                 dbpath_source, dbpath_log, dbpath_errorlog):
        self.collection = collection                    # 爬蟲任務配置（每個 col 對應的設定）
        self.log = log                                  # 任務執行狀態記錄
        self.errorlog = errorlog                        # 詳細錯誤紀錄
        self.dbpath_source = dbpath_source              # 資料儲存路徑
        self.dbpath_log = dbpath_log                    # log 儲存路徑
        self.dbpath_errorlog = dbpath_errorlog          # error log 儲存路徑
        self.request_module = requests                  # HTTP 請求模組
        self.randomheader = randomheader                # 隨機 header 函式
        self.picklesave = picklesave                    # 儲存函式
        self.sleepteller = sleepteller                  # sleep 控制函式

# === 執行任務主體 ===
class CrawlerTask:
    """
    執行單一筆爬蟲任務，並處理資料儲存、錯誤處理與 log 記錄。
    """
    def __init__(self, config: CrawlerConfig):
        self.cfg = config

    def run_task(self, ind, col):
        # 依照日期與欄位名稱取得爬蟲設定
        crawlerdic = self.cfg.collection[col]
        crawlerdic['payload']['date'] = ind

        # 建立儲存目錄
        datapath = join(self.cfg.dbpath_source, col)
        makedirs(datapath, exist_ok=True)
        # print("first")
        # print(type(ind))
        print(ind, col)

        try:
            # 發送 POST 請求
            res = self.cfg.request_module.post(
                url=crawlerdic['url'],
                headers=next(self.cfg.randomheader()),
                data=crawlerdic['payload'],
                timeout=(3, 7)
            )
            # 嘗試解析 JSON（可能會因反爬被攔截）
            data = res.json()
            print('sleep ...')
            self.cfg.sleepteller()

        except KeyboardInterrupt:
            # 人為中斷時儲存資料後退出
            self._handle_interrupt()

        except Exception as e:
            # 任意錯誤（連線失敗、json 解析錯、反爬頁面等）
            self._handle_request_error(ind, col, crawlerdic, e)
            return

        # 伺服器回應正常（200 OK）
        if res.status_code == self.cfg.request_module.codes.ok:
            # stat網頁回應是ok，還有回應確實有拿到資料，data不能是空的，這樣才是正確的回應，如果只是連線ok但資料是空的，就代表那天放假
            # 但是data的key有可能叫做data1、data2所以不能直接用data去判斷，會因為key的名稱產生誤判
            if data.get('stat') == 'OK':
                self.cfg.log.loc[self.cfg.log.index == ind, col] = 'succeed'

                print(type(ind))
                print(ind)
                # print(pd.to_datetime(ind))
            else:
                self.cfg.log.loc[self.cfg.log.index == ind, col] = 'close'
                print(type(ind))
                print(ind)
                # print(pd.to_datetime(ind))
                self.cfg.picklesave(self.cfg.log, self.cfg.dbpath_log)
                return
        else:
            # 非 200 回應（如 403、500），需記錄錯誤
            self._handle_status_error(ind, col, crawlerdic, res)
            sys.exit(0)

        # 組裝資料儲存
        data['crawlerdic'] = crawlerdic
        data['request'] = res
        filename = join(datapath, f"{col}_{ind.date()}.pkl")
        self.cfg.picklesave(data, filename)

        # 若資料頻率是月頻，清除當月舊檔案
        if crawlerdic.get('freq') == 'M':
            self._clean_monthly_files(ind, col, datapath)

        # 儲存更新後 log
        self.cfg.picklesave(self.cfg.log, self.cfg.dbpath_log)

    # === 中斷處理 ===
    def _handle_interrupt(self):
        print("KeyboardInterrupt ... content saving")
        self.cfg.picklesave(self.cfg.log, self.cfg.dbpath_log)
        self.cfg.picklesave(self.cfg.errorlog, self.cfg.dbpath_errorlog)
        print("Log saved .")
        sys.exit()

    # === 請求過程出錯（無回應、json fail） ===
    def _handle_request_error(self, ind, col, crawlerdic, e):
        print("Unknowned error")
        print("===============")
        print(format_exc())
        print("===============")
        print(e)
        print(type(ind))
        print(ind)
        print(pd.to_datetime(ind))
        self.cfg.log.loc[self.cfg.log.index == ind, col] = 'request error'
        print(ind)
        errordic = {
            'crawlerdic': crawlerdic,
            'errormessage1': format_exc(),
            'errormessage2': e,
            'errormessage3': 'request failed'
        }
        self.cfg.errorlog.loc[ind, col] = [errordic]
        self.cfg.picklesave(self.cfg.log, self.cfg.dbpath_log)
        self.cfg.picklesave(self.cfg.errorlog, self.cfg.dbpath_errorlog)
        self.cfg.sleepteller(mode='long')

    # === 回應狀態錯誤（如 403, 500） ===
    def _handle_status_error(self, ind, col, crawlerdic, res):
        print("Unknowned error")
        print("===============")
        print(type(ind))
        print(ind)
        print(pd.to_datetime(ind))
        self.cfg.log.loc[self.cfg.log.index == ind, col] = 'result error'
        errordic = {
            'crawlerdic': crawlerdic,
            'request': res,
            'requeststatus': res.status_code,
            'errormessage1': 'result error'
        }
        self.cfg.errorlog.loc[ind, col] = [errordic]
        self.cfg.picklesave(self.cfg.log, self.cfg.dbpath_log)
        self.cfg.picklesave(self.cfg.errorlog, self.cfg.dbpath_errorlog)
        self.cfg.sleepteller(mode='long')

    # === 月頻資料：只留當天，刪除當月過往 ===
    def _clean_monthly_files(self, ind, col, datapath):
        daterange = pd.date_range(
            start=ind.strftime('%Y-%m-1'),
            end=ind - datetime.timedelta(days=1),
            freq='D',
            inclusive='left'
        )
        for d in daterange:
            oldfile = join(datapath, f"{col}_{d}.pkl")
            if exists(oldfile):
                remove(oldfile)

# === 主控制邏輯 ===
def main(collection, log, errorlog, dbpath_source, dbpath_log, dbpath_errorlog):
    """
    建立配置與任務執行器，批次執行所有待處理的資料。
    """
    config = CrawlerConfig(
        collection=collection,
        log=log,
        errorlog=errorlog,
        dbpath_source=dbpath_source,
        dbpath_log=dbpath_log,
        dbpath_errorlog=dbpath_errorlog
    )
    crawler = CrawlerTask(config)

    for ind, col in findval(log, 'wait'):
        crawler.run_task(ind, col)

# === 可執行介面 ===
if __name__ == '__main__':
    # 這裡可以匯入資料後執行 main(...)
    # 範例：
    # main(collection, log, errorlog, 'data/', 'data/log.pkl', 'data/errorlog.pkl')
    pass
