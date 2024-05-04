#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:13:14 2020

@author: stevenhsu
"""
import pandas as pd
from os.path import join, exists
from StevenTricks.snt import findbylist
from StevenTricks.fileop import PathWalk_df, pickleload, logfromfolder, picklesave
from StevenTricks.dfi import findval
from StevenTricks.warren.twse import Log
from StevenTricks.warren.conf import db_path, colname_dic, numericol, collection, dropcol, datecol
from StevenTricks.dbsqlite import tosql_df, readsql_iter
from StevenTricks.snt import tonumeric_int, changetype_stringtodate


productkey = {
    "col": ["field"],
    "value": ["data", "list"],
    "title": ["title"]
        }


def getkeys(data):
    productcol = {
        "col": [],
        "value": [],
        "title": [],
        }
    for key in sorted(data.keys()):
        for k, i in productkey.items():
            i = [key for _ in i if _ in key.lower()]
            if i:
                productcol[k] += i
    return pd.DataFrame(productcol)


def productdict(source, key):
    productdict = {}
    for col, value, title in key.values:
        if not source[value]:
            continue
        df = pd.DataFrame(data=source[value], columns=source[col])
        productdict[source[title]] = df
    return productdict


def type1(df, title, subtitle):
    df = df.replace({",": "", r'\)': '', r'\(': '_'}, regex=True)
    df = df.rename(columns=colname_dic)
    df = df.drop(columns=dropcol, errors='ignore')
    df.loc[:, numericol[title][subtitle]] = df[numericol[title][subtitle]].apply(pd.to_numeric, errors='coerce')
    return {subtitle: df}


def type2(df, title, subtitle):
    # 處理xxx(yy)的格式，可以把xxx和(yy)分開成上下兩列
    res = []
    for subcol in df:
        # 原本的col name因為從0又重新賦值成subcol，所以原本的col不變，但是新分出來的col，全部都命名成1，並且全部放到res
        res.append(df[subcol].str.split(r'\(', expand=True, regex=True).rename(columns={0: subcol}))
    # 從res把剛剛分出來的重新組回來，就會變成只要是剛剛從括弧分出來的欄位名稱都會是1
    res = pd.concat(res, axis=1)
    # 只要把欄位名稱是1的全部丟掉，就又會回到原本的df
    df = res.drop(1, axis=1)
    # 把欄位名稱是1的全部選出來就會是純新的表格
    res = res.loc[:, 1]
    # 在把原本的column全部放到新的表格
    res.columns = df.columns
    # 在用concat上下組合起來
    df = pd.concat([df, res], ignore_index=True).dropna()
    # df = df.replace({r'\)': ''}, regex=True)
    df = type1(df, title=title, subtitle=subtitle)
    return df


def type3(df, title, subtitle):
    # 有交易單位的另外新增一個叫做'單位'的欄位
    res = []
    for subcol in df:
        # 原本的col name因為從0又重新賦值成subcol，所以原本的col不變，但是新分出來的col，全部都命名成unit，預設應該只會有一個欄位被命名成unit
        res.append(df[subcol].str.split(r'\(', expand=True, regex=True).rename(columns={0: subcol, 1: 'unit'}))
    # 從res把剛剛分出來的重新組回來，concat如果加ignore_index，那就會連column name 都變成連續數字，所以不能加
    df = pd.concat(res, axis=1).dropna()
    # 剛剛沒被清到的右括弧重新清理
    # df = df.replace({r'\)': ''}, regex=True)
    df = type1(df, title=title, subtitle=subtitle)
    return df


def type4(df, title, subtitle):
    df = type1(df, title=title, subtitle=subtitle)
    df = df[subtitle]
    df.columns = ",".join(df.columns).replace("買進", "融券買進").replace("融券買進", "融資買進", 1).split(",")
    df.columns = ",".join(df.columns).replace("賣出", "融券賣出").replace("融券賣出", "融資賣出", 1).split(",")
    df.columns = ",".join(df.columns).replace("今日餘額", "今日融券餘額").replace("今日融券餘額", "今日融資餘額", 1).split(",")
    df.columns = ",".join(df.columns).replace("限額", "融券限額").replace("融券限額", "融資限額", 1).split(",")
    return {subtitle: df}


def type5(df, title, subtitle):
    df = type1(df, title=title, subtitle=subtitle)
    df = df[subtitle]
    df = changetype_stringtodate(df=df, datecol=['date'], mode=4)
    return {subtitle: df}


def type6(df, title, subtitle):
    df = type1(df, title=title, subtitle=subtitle)
    df = df[subtitle]
    df.columns = ",".join(df.columns).replace("前日餘額", "借券前日餘額").replace("借券前日餘額", "融券前日餘額", 1).split(",")
    return {subtitle: df}


def type7(df, title, subtitle):
    df.columns = ",".join(df.columns).replace(r"</br>", "").split(",")
    df = type1(df, title=title, subtitle=subtitle)
    # 因為最後返回的結果是用type1的，所以直接返回df
    return df


def type8(df, title, subtitle):
    df = type1(df, title=title, subtitle=subtitle)
    df = df[subtitle]
    df = changetype_stringtodate(df=df, datecol=['最近一次上市公司申報外資持股異動日期'], mode=4)
    return {subtitle: df}


fundic = {
    '每日收盤行情': {
        '價格指數(臺灣證券交易所)': type1,
        '價格指數(跨市場)': type1,
        '價格指數(臺灣指數公司)': type1,
        '報酬指數(臺灣證券交易所)': type1,
        '報酬指數(跨市場)': type1,
        '報酬指數(臺灣指數公司)': type1,
        '大盤統計資訊': type1,
        '漲跌證券數合計': type2,
        '每日收盤行情': type1,
    },
    '信用交易統計': {
        "融資融券彙總": type4,
        "信用交易統計": type3,
    },
    '市場成交資訊': {
        '市場成交資訊': type5
    },
    '三大法人買賣金額統計表': {
        '三大法人買賣金額統計表': type1
    },
    '三大法人買賣超日報': {
        '三大法人買賣超日報': type1
    },
    '個股日本益比、殖利率及股價淨值比': {
        '個股日本益比、殖利率及股價淨值比': type1
    },
    '信用額度總量管制餘額表': {
        '信用額度總量管制餘額表': type6
    },
    '當日沖銷交易標的及成交量值': {
        '當日沖銷交易統計資訊': type7,
        '當日沖銷交易標的及成交量值': type7,
    },
    "每月當日沖銷交易標的及統計": {
        "每月當日沖銷交易標的及統計": type5,
    },
    '外資及陸資投資持股統計': {
        '外資及陸資投資持股統計': type8,
    },
    '發行量加權股價指數歷史資料': {
        '發行量加權股價指數歷史資料': type5
    },
}


def cleaner(product, title):
    # data 就是直接讀取pkl檔案得到的data
    # title就是大標，pkl檔案裡面有subtitle小標
    # pkl的小標資料不乾淨，需要透過轉換，所以就有find
    # 返回的資料會是dict{subtitle:df}
    res = {}
    for key, df in product.items():
        find = findbylist(collection[title]['subtitle'], key)
        # 把小標做轉換成find
        if find:
            if len(find) > 1:
                print('{} is in {} at the same time.'.format(key, ','.join(find)))
                break
            else:
                # print(find[0], title, df)
                fun = fundic[title][find[0]]
                res.update(fun(df, title, find[0]))
        else:
            print('{} is not in crawlerdic.SubItem.'.format(key))
            break
    return res


if __name__ == '__main__':
    # a=pickleload(r'/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/source/stocklist/2/股票/股票_2023-01-27.pkl')
    stocklog = Log(db_path)
    # 初始化
    log = stocklog.findlog('source', 'log.pkl')
    # 讀取log
    log = logfromfolder(join(db_path, 'source'), fileinclude=['.pkl'], fileexclude=['log'], direxclude=['stocklist'], dirinclude=[], log=log, fillval='succeed', avoid=['cleaned'])
    # 整理log，如果要把db刪掉重新進行資料清理，就要把avoid重新設置成空的[]，不然他會跳過cleaned，要重新進行資料清理就要把他設置成succeed
    log_stocklist_path = join(db_path, 'source', 'stocklistlog.pkl')
    # 設定log_stocklist的路徑
    # 整理stocklist的log
    if exists(log_stocklist_path) is True:
        log_stocklist = pickleload(path=log_stocklist_path)
        log_stocklist = logfromfolder(join(db_path, 'source', 'stocklist'), fileinclude=['.pkl'], fileexclude=['log'],
                                      dirinclude=['stocklist'], direxclude=[], log=log_stocklist, fillval='succeed', avoid=['cleaned'])
    else:
        log_stocklist = logfromfolder(join(db_path, 'source', 'stocklist'), fileinclude=['.pkl'], fileexclude=['log'], dirinclude=['stocklist'], direxclude=[], log=pd.DataFrame(), fillval='succeed')
    # 讀取stocklist的log
    files = PathWalk_df(path=join(db_path, 'source'), direxclude=['stocklist'], fileexclude=['log'], fileinclude=['.pkl'])
    # 一般檔案的path
    files_stocklist = PathWalk_df(path=join(db_path, 'source'), dirinclude=['stocklist'], fileexclude=['log'], fileinclude=['.pkl'])
    # stocklist的檔案path

    # n=0
    for ind, col in findval(log_stocklist, 'succeed'):
        # print(ind, col)
        # if n==3:break
        # n+=1
        # 用succeed當條件
        data = pickleload(files_stocklist.loc[files_stocklist['file'] == '{}_{}.pkl'.format(col, ind), 'path'].values[0])
        # 讀取檔案當作data
        if data.empty is True:
            # 有些下載下來本身就是空值，要做特殊處理，直接跳過，但是要做log紀錄
            log_stocklist.loc[ind, col] = 'cleaned'
            picklesave(data=log_stocklist, path=log_stocklist_path)
            continue
        data = data.rename(columns=colname_dic)
        # 開始欄位rename
        for key in ['指數代號及名稱', '有價證券代號及名稱']:
            # 名稱欄位要把代號和名稱拆開成兩欄
            if key in data :
                data.loc[:, ['代號', '名稱']] = data[key].str.split(r'\u3000', expand=True).rename(columns={0: '代號', 1: '名稱'})
                # \u3000是全形的空白鍵，就算前面不加r也能判斷成功，但怕以後會不能用｜去做多重判斷，所以先放r
                data = data.drop(key, axis=1)
                # 最後要drop本來的key
        # 以上之後可以考慮做成function
        if tonumeric_int(col[-1]) is not None:
            col = col[:-1]
        #     把尾數的數字篩選掉
        colrename = colname_dic.get(col, col)
        # 欄位的rename
        data.loc[:, ['type', 'date']] = colrename, ind
        # 新增兩個欄位
        data.loc[:, [_ for _ in numericol['stocklist'] if _ in data]] = data[[_ for _ in numericol['stocklist'] if _ in data]].apply(pd.to_numeric, errors='coerce')
        # 利率值是空的就代表是浮動利率
        data.loc[:, [_ for _ in datecol['stocklist'] if _ in data]] = data[[_ for _ in datecol['stocklist'] if _ in data]].apply(pd.to_datetime, errors='coerce')
        # 到期日，日期是空的就代表無到期日
        tosql_df(df=data, dbpath=join(db_path, 'cleaned', 'stocklist.db'), table=colrename, pk=["ISINCode"])
        # 放進db，用最簡單的模式，直覺型放入，沒有用adapter
        log_stocklist.loc[ind,col] = 'cleaned'
        # 成功放進db之後就要改成cleaned
        picklesave(data=log_stocklist, path=log_stocklist_path)
        # 儲存log

    stocklist = pd.concat(readsql_iter(dbpath=join(db_path, 'cleaned', 'stocklist.db')))
    # 讀取stocklist，以利下面可以merge
    # n = 1
    for ind, col in findval(log, 'succeed'):
    # for ind, col in findval(log.drop(['每日收盤行情', '市場成交資訊', '信用交易統計', '三大法人買賣超日報', '三大法人買賣金額統計表', '個股日本益比、殖利率及股價淨值比', '當日沖銷交易標的及成交量值','信用額度總量管制餘額表',"每月當日沖銷交易標的及統計", '外資及陸資投資持股統計'], axis=1), 'succeed'):
        print(col, ind)
        # if n == 10:
        #     break
        # n += 1
        file = pickleload(files.loc[files['file'] == '{}_{}.pkl'.format(col, ind.date()), 'path'].values[0])
        # 讀取pkl檔案
        keydf = getkeys(file)
        # 找到所有key對應的資料
        product = productdict(source=file, key=keydf)
        # 把key對應的結果和product合併起來
        res = cleaner(product=product, title=col)
        # 清理結果要取出
        for key, df in res.items():
            # nomatch就是不用跟stocklist進行配對
            if key not in collection[col]['nomatch']:
                # merge就是優先用代號，沒有代號就用名稱
                if '代號' in df:
                    df = df.merge(stocklist.loc[:, [_ for _ in stocklist if _ not in df.drop('代號', axis=1)]], how='left', on=['代號'])
                else:
                    df = df.merge(stocklist.loc[:, [_ for _ in stocklist if _ not in df.drop('名稱', axis=1)]], how='left', on=['名稱'])
            # 這裡決定如何用pk，優先用代號，再來是名稱，最後是空的，就是直接給他用auto_pk
            if '代號' in df:
                pk = ['代號', 'date']
            elif '名稱' in df:
                pk = ['名稱', 'date']
            else:
                pk = []

            key = colname_dic.get(key, key)
            # key的轉換主要是把括號弄掉和一些常用字的轉換
            if 'date' not in df:
                df.loc[:, ['date']] = ind
            # 全部都要新增日期，就算有merge，這裡也要把stocklist裏面的date覆蓋掉，table就是等一下放盡sqldb要用的table name

            tosql_df(df=df, dbpath=join(db_path, 'cleaned', '{}.db'.format(ind.year)), table=key, pk=pk)

        log.loc[ind, col] = 'cleaned'
        picklesave(log, join(db_path, 'source', 'log.pkl'))
