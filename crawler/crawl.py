#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:22:32 2020

@author: mac
"""

from StevenTricks.dfi import findval, periodictable
from StevenTricks.netGEN import randomheader
from StevenTricks.file_utils import logfromfolder, logmaker, picklesave, pickleload, sweep_path, PathWalk_df
from StevenTricks.process import sleepteller
from conf import collection, dailycollection, path_dic
from crawler.schema_utils import warehouseinit
from os import remove, makedirs
from os.path import join, exists
from traceback import format_exc

import sys
import requests as re
import pandas as pd
import datetime

dbpath = path_dic["stock_twse_db"]
dbpath_source = join(dbpath, "source")
dbpath_log = join(dbpath_source, "log", "log.pkl")
dbpath_errorlog = join(dbpath_source, "log", "errorlog.pkl")
log_info = sweep_path(dbpath_log)
errorlog_info = sweep_path(dbpath_errorlog)
if __name__ == "__main__":
    warehouseinit(dbpath)
    # 每一次被當成主要模組呼叫，都會自動生成倉庫資料夾，為了確保一定有資料夾，所以每次使用都要呼叫一次
    # 先判斷有沒有log
    if log_info["exists"] is True:
        log = pickleload(dbpath_log)
        # 有log還要判斷是不是最新的
        if datetime.date.today() not in log.index:
            print("{} not in log index, updating the log table".format(str(datetime.date.today())))
            latest_log = periodictable(collection, datemin=log.index.max()+datetime.timedelta(days=1))
            # 從上一次創建log的最新天數開始，所以要加一天，然後開始創建新的table
            log = pd.concat([log, latest_log])
    else:
        log = periodictable(collection)
    # 不管有沒有log，在爬蟲啟動之前都會根據目前資料夾的資料來更新log，確保抓取沒有遺漏
    # 再判斷有沒有errorlog
    if errorlog_info["exists"] is True:
        errorlog = pickleload(dbpath_errorlog)
    else:
        errorlog = pd.DataFrame()
    print("LOG讀取成功")
    print("開始更新LOG")
    # 先盤點資料
    dbpath_lis = PathWalk_df(dbpath_source, [], ["log"], [], [])
    # 再更新log檔
    log = logfromfolder(dbpath_lis, log=log, fillval='succeed')
    # 比對資料夾內的資料，依照現有存在的資料去比對比較準確，有可能上次抓完，中間有動到資料
    print("Log檔案更新結束\nLog程序處理結束")
    # log處理結束
    print("開始進行網路資料下載")

    for _ in dailycollection['stocklist']['modelis']:
        df = pd.read_html(dailycollection['stocklist']['url'].format(str(_)), encoding='cp950')
        sleepteller()
        df = pd.DataFrame(df[0])
        # read_html是返回list，所以要取出0
        if df.empty is True:
            # 代表什麼都沒返回
            print("stocktable No:{} ___empty crawled result".format(str(_)))
            continue
        df.columns = df.loc[0]
        # 指定第一列為column
        df = df.drop(0)
        # 避免跟column重複，所以先刪掉
        df = df.reset_index(drop=True).reset_index()
        # 先弄出一列是連續數字出來
        tablename = [list(set(_)) for _ in df.values if len(set(_)) == 2]
        # 要找出一整列都是重複的，當作table name，因為剛剛已經用reset_index用出一整數列了，得出的重複值會長這樣[3,重複值]，所以如果是我們要找的重複值，最少會有兩個值，一個是數列，一個是重複值
        df = df.drop(["index", "Unnamed: 6"], errors="ignore", axis=1)
        # 把index先刪掉也把用不到的數列先刪掉，欄位清理

        # tablename
        if len(tablename) > 1:
            # 如果有兩種以上產品名的情況
            name_index = [(a, b) for a, b in zip(tablename, tablename[1:] + [[None]])]
        elif len(tablename) == 1:
            # 只有一種產品名的情況
            name_index = [(tablename[0], [None])]
        else:
            # 沒有產品名，就單獨一個dataframe，那就不用特別處理直接儲存就好
            table = "無細項分類的商品{}".format(str(_))
            datapath = join(dbpath_source, 'stocklist', table, datetime.datetime.today().strftime(table+'_%Y-%m-%d.pkl'))
            picklesave(df, datapath)
            continue
        # 利用同一個row的重複值來判斷商品項目名稱，同時判斷儲存的方式
        # 對於有產品細項名稱的商品開始做以下特殊處理
        for nameindex in name_index:
            start = nameindex[0]
            end = nameindex[1]

            startname, startint = [_ for _ in start if isinstance(_, str) is True][0], [_ for _ in start if isinstance(_, str) is False][0]
            endint = [_ for _ in end if isinstance(_, str) is False][0]
            # 先抓出起始的值和尾端值然後用slice來做切割，把資料分段儲存進table

            if end[0] is None:
                df_sub = df[startint + 1:]
            else:
                df_sub = df[startint + 1:endint]

            datapath = join(dbpath_source, 'stocklist', startname, datetime.datetime.today().strftime(startname + str(_) + '_%Y-%m-%d.pkl'))
            picklesave(df_sub, datapath)

    # for ind, col in findval(log.drop(['每日收盤行情', '信用交易統計', '市場成交資訊', '三大法人買賣金額統計表', '三大法人買賣超日報', '個股日本益比、殖利率及股價淨值比', '信用額度總量管制餘額表', '當日沖銷交易標的及成交量值', "每月當日沖銷交易標的及統計", '外資及陸資投資持股統計'], axis=1), 'wait'):
    for ind, col in findval(log, 'wait'):
        crawlerdic = collection[col]
        crawlerdic['payload']['date'] = ind.date().strftime("%Y%m%d")
        datapath = join(dbpath_source, col)
        print(ind, col)
        makedirs(datapath, exist_ok=True)

        try:
            res = re.post(url=crawlerdic['url'], headers=next(randomheader()), data=crawlerdic['payload'], timeout=(3, 7))
            # 有時候會出現回應ok，回應是200的狀況，但是json()會出現error，這樣也是當作錯誤，這是因為網路被反爬蟲的關係
            data = res.json()
            print('sleep ...')
            sleepteller()
        except KeyboardInterrupt:
            print("KeyboardInterrupt ... content saving")
            picklesave(log,dbpath_log)
            picklesave(log,dbpath_errorlog)
            print("Log saved .")
            sys.exit()
        except Exception as e:
            print("Unknowned error")
            print("===============")
            print(format_exc())
            # 較詳細的錯誤訊息
            print("===============")
            print(e)
            # 較簡陋的錯誤訊息
            log.loc[log.index == ind, col] = 'request error'
            errordic = {'crawlerdic': crawlerdic,
                        'errormessage1': format_exc(),
                        'errormessage2': e,
                        'errormessage3': 'request failed'}
            errorlog.loc[ind, col] = [errordic]
            picklesave(log, dbpath_log)
            picklesave(log, dbpath_errorlog)
            sleepteller(mode='long')
            continue

        if res.status_code == re.codes.ok:
            # 只要result的結果是正確，且json()又不出錯，大概就一定有正確資料，就只剩下是有資料還是當天休市的差別
            print(data['stat'])
            print('------------------')
            if data['stat'] == 'OK':
                log.loc[log.index == ind, col] = 'succeed'
            else:
                # 例假日或颱風假
                log.loc[log.index == ind, col] = 'close'
                picklesave(log, dbpath_log)
                continue
        else:
            print("Unknowned error")
            print("===============")
            log.loc[log.index == ind, col] = 'result error'
            errordic = {'crawlerdic': crawlerdic,
                        'request': res,
                        'requeststatus': res.status_code,
                        'errormessage1': 'result error'}
            errorlog.loc[ind, col] = [errordic]
            picklesave(log, dbpath_log)
            picklesave(log, dbpath_errorlog)
            sleepteller(mode='long')
            # continue
            exit(0)
        data['crawlerdic'] = crawlerdic
        data['request'] = res
        picklesave(data=data, path=join(datapath, col+'_'+str(ind.date()))+'.pkl')

        # 把以月為頻率的資料要刪除之前的資料，留當月最新的就好，不用每天都留
        if crawlerdic['freq'] == 'M':
            daterange = pd.date_range(start=ind.strftime('%Y-%m-1'), end=ind-datetime.timedelta(days=1), freq='D', inclusive='left')
            for d in daterange:
                if exists(join(datapath, col+'_'+str(d))+'.pkl'):
                    remove(join(datapath, col+'_'+str(d))+'.pkl')

        picklesave(log, dbpath_log)
