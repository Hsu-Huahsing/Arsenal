from os.path import join


#因為爬蟲的結果會有很多db要儲存，要先指定warehouse(倉庫)的路徑
path_dic = {
    "stock_twse_db": r"/Users/stevenhsu/Library/Mobile Documents/com~apple~CloudDocs/warehouse/stock/twse"
}

dbpath = path_dic["stock_twse_db"]
dbpath_source = join(dbpath, "source")
dbpath_cleaned = join(dbpath, "cleaned")
dbpath_cleaned_log = join(dbpath_cleaned,"log.pkl")
dbpath_log = join(dbpath_source, "log", "log.pkl")
dbpath_errorlog = join(dbpath_source, "log", "errorlog.pkl")
dbpath_productlist = join(dbpath_source, "productlist.pkl")

product_clean = {
    "code" : {
        1 : [5,9],
        2 : [5,9],
        3 : [5,10],
        4 : [5,11],
        5 : [5,9],
        6 : [5,10],
        7 : [5,11],
        8 : [5,9],
        9 : [5,11],
        10 : [5,9],
        11 : [5,11],
        12 : [5,11],

    }
}


#先統一全部爬蟲抓下來的table使用的column name
colname_dic = {
    "指數代號及名稱": "名稱",
    "STO代號及名稱": "名稱",
    "每日收盤行情(全部)": "每日收盤行情",
    "公開發行日": "發行日",
    "上市日": "發行日",
    "登錄日": "發行日",
    "掛牌日": "發行日",
    "價格指數(跨市場)": "價格指數_跨市場",
    "價格指數(臺灣指數公司)": '價格指數_臺灣指數公司',
    "報酬指數(臺灣證券交易所)": '報酬指數_臺灣證券交易所',
    "報酬指數(跨市場)": '報酬指數_跨市場',
    "報酬指數(臺灣指數公司)": '報酬指數_臺灣指數公司',
    '價格指數(臺灣證券交易所)': '價格指數_臺灣證券交易所',
    "上市認購(售)權證": "上市認購售權證",
    "上櫃認購(售)權證": "上櫃認購售權證",
    '認購(售)權證': "認購售權證",
    "臺灣存託憑證(TDR)": "台灣存託憑證",
    "受益證券-不動產投資信託": "受益證券_不動產投資信託",
    "國際證券辨識號碼(ISIN Code)": "ISINCode",
    "受益證券-資產基礎證券": "受益證券_資產基礎證券",
    "黃金期貨(USD)": "黃金期貨USD",
    "成交金額(元)": "成交金額_元",
    "成交股數(股)": "成交股數_股",
    "漲跌百分比(%)": "漲跌百分比%",
    "自營商買進股數(自行買賣)": "自營商買進股數_自行買賣",
    "自營商賣出股數(自行買賣)": "自營商賣出股數_自行買賣",
    "自營商買賣超股數(自行買賣)": "自營商買賣超股數_自行買賣",
    "自營商買進股數(避險)": "自營商買進股數_避險",
    "自營商賣出股數(避險)": "自營商賣出股數_避險",
    "自營商買賣超股數(避險)": "自營商買賣超股數_避險",
    "殖利率(%)": "殖利率%",
    "外陸資買進股數(不含外資自營商)": "外陸資買進股數_不含外資自營商",
    "外陸資賣出股數(不含外資自營商)": "外陸資賣出股數_不含外資自營商",
    "外陸資買賣超股數(不含外資自營商)": "外陸資買賣超股數_不含外資自營商",
    "現金(券)償還": "現金券償還",
    "證券代號": "代號",
    "股票代號": "代號",
    "指數代號": "代號",
    "證券名稱": "名稱",
    "股票名稱": "名稱",
    "有價證券名稱": "名稱",
    "有價證券代號及名稱": "名稱",
    '報酬指數': "名稱",
    '指數': "名稱",
    '成交統計': "名稱",
    '類型': "名稱",
    '項目': "名稱",
    '單位名稱': "名稱",
    '日期': 'date',
    "融資(交易單位)" : "融資_交易單位",
    "融券(交易單位)" : "融券_交易單位",
    "融資金額(仟元)" : "融資金額仟元",
}

#這裡是使用在爬蟲裡面request的header
headers = {
    'mac': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Safari/605.1.15',
    'safari14.0': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'iphone13': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_1_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.1 Mobile/15E148 Safari/604.1',
    'ipod13': 'Mozilla/5.0 (iPod; CPU iPhone OS 13_1_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.1 Mobile/15E148 Safari/604.1',
    'ipadmini13': 'Mozilla/5.0 (iPad; CPU iPhone OS 13_1_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.1 Mobile/15E148 Safari/604.1',
    'ipad': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.1 Safari/605.1.15',
    'winedge': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299',
    'chromewin': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.70 Safari/537.36',
    'firefoxmac': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:70.0) Gecko/20100101 Firefox/70.0',
    'firefoxwin': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0'
}

key_set = {
    "main1" : {
        "fields":["fields"],
        "data":["data"],
        "title":["title","subtitle"],
        "groups":["groups"],
        "notes":["notes"],
    },
    "set1" : {
        "fields":["creditFields"],
        "data":["creditList"],
        "title":["creditTitle"],
        "notes":["creditNotes"],
    }
}

fields_span = {
    "融資融券彙總" : {
        "股票" : {
            "start" : 0,
            "end" : 2,
        },
        "融資" : {
            "start" : 2,
            "end" : 8,
        },
        "融券" : {
            "start" : 8,
            "end" : 14,
        },
        "其他" : {
            "start" : 14,
            "end" : 16,
        },
    },
    "信用額度總量管制餘額表" : {
        "股票": {
            "start": 0,
            "end": 2,
        },
        "融券": {
            "start": 2,
            "end": 8,
        },
        "借券賣出": {
            "start": 8,
            "end": 14,
        },
        "其他": {
            "start": 14,
            "end": 15,
        },
    }
}

transtonew_col = {
    '信用額度總量管制餘額表': {
        '信用額度總量管制餘額表': {
            "借券賣出_賣出":'借券賣出_當日賣出',
            "借券賣出_庫存異動":'借券賣出_當日還券',
            "借券賣出_今日餘額":'借券賣出_當日餘額',
            "借券賣出_可使用額度":'借券賣出_次一營業日可限額'
        },
    },
    '三大法人買賣超日報': {
        '三大法人買賣超日報':{
            '外資買進股數' : "外陸資買進股數_不含外資自營商",
            '外資賣出股數' :"外陸資賣出股數_不含外資自營商",
            '外資買賣超股數':"外陸資買賣超股數_不含外資自營商",
            "自營商買進股數":"自營商買進股數_自行買賣",
            "自營商賣出股數":"自營商賣出股數_自行買賣",
        }
    },
}

# 這裡給的col欄位都是要清理過後的
numericol = {
    'stocklist': ['利率值'],
    '每日收盤行情': {
        "每日收盤行情": ['成交股數', '成交筆數', '成交金額', '開盤價', '最高價', '最低價', '收盤價', '漲跌價差', '最後揭示買價', '最後揭示買量', '最後揭示賣價', '最後揭示賣量', '本益比'],
        '報酬指數_臺灣證券交易所': ['收盤指數', '漲跌點數', '漲跌百分比%'],
        '報酬指數_臺灣指數公司': ['收盤指數', '漲跌點數', '漲跌百分比%'],
        "價格指數_臺灣證券交易所": ['收盤指數', '漲跌點數', '漲跌百分比%'],
        '大盤統計資訊': ["成交金額_元", "成交股數_股", '成交筆數'],
        '價格指數_跨市場': ['收盤指數', '漲跌點數', '漲跌百分比%'],
        '價格指數_臺灣指數公司': ['收盤指數', '漲跌點數', '漲跌百分比%'],
        '報酬指數_跨市場': ['收盤指數', '漲跌點數', '漲跌百分比%'],
    },
    "信用交易統計": {
        "融資融券彙總": ['資券互抵','融資_買進', '融資_賣出', '融資_現金償還', '融資_前日餘額', '融資_今日餘額',"融資_次一營業日餘額", '融資_限額', '融券_現券償還','融券_買進', '融券_賣出', '融券_前日餘額', '融券_今日餘額', '融券_限額',"融券_次一營業日餘額"],
        "信用交易統計": ['買進', '賣出', '現金券償還', '前日餘額', '今日餘額'],
    },
    '市場成交資訊': {
        '市場成交資訊': ['成交股數', '成交金額', '成交筆數', '發行量加權股價指數', '漲跌點數']
    },
    '三大法人買賣金額統計表': {
        '三大法人買賣金額統計表': ['買進金額', '賣出金額', '買賣差額']
    },
    '三大法人買賣超日報': {
        '三大法人買賣超日報': ["外陸資買進股數_不含外資自營商","外陸資賣出股數_不含外資自營商","外陸資買賣超股數_不含外資自營商","外資自營商買進股數","外資自營商賣出股數","外資自營商買賣超股數",'外資買進股數', '外資賣出股數', '外資買賣超股數', '投信買進股數', '投信賣出股數', '投信買賣超股數', '自營商買進股數', '自營商賣出股數', '自營商買賣超股數', '三大法人買賣超股數',"自營商買進股數_自行買賣","自營商賣出股數_自行買賣","自營商買賣超股數_自行買賣","自營商買進股數_避險","自營商賣出股數_避險","自營商買賣超股數_避險"]
    },
    '個股日本益比、殖利率及股價淨值比': {
        '個股日本益比、殖利率及股價淨值比': ['本益比', '殖利率%', '股價淨值比']
    },
    '信用額度總量管制餘額表': {
        '信用額度總量管制餘額表': ['融券_前日餘額', '融券_賣出', '融券_買進', '融券_現券', '融券_今日餘額', '融券_限額', '融券_次一營業日限額','借券賣出_前日餘額', '借券賣出_當日賣出', '借券賣出_當日還券', '借券賣出_當日調整', '借券賣出_當日餘額', '借券賣出_次一營業日可限額',"借券賣出_賣出","借券賣出_庫存異動","借券賣出_今日餘額","借券賣出_可使用額度"]
    },
    '當日沖銷交易標的及成交量值': {
        '當日沖銷交易統計資訊': ['當日沖銷交易總成交股數', '當日沖銷交易總成交股數占市場比重%', '當日沖銷交易總買進成交金額', '當日沖銷交易總買進成交金額占市場比重%', '當日沖銷交易總賣出成交金額', '當日沖銷交易總賣出成交金額占市場比重%'],
        '當日沖銷交易標的及成交量值': ['當日沖銷交易成交股數', '當日沖銷交易買進成交金額', '當日沖銷交易賣出成交金額'],
    },
    '每月當日沖銷交易標的及統計': {
        '每月當日沖銷交易標的及統計': ['當日沖銷交易總成交股數', '當日沖銷交易總成交股數占市場比重%', '當日沖銷交易總買進成交金額', '當日沖銷交易總買進成交金額占市場比重%', '當日沖銷交易總賣出成交金額', '當日沖銷交易總賣出成交金額占市場比重%']
    },
    '外資及陸資投資持股統計': {
        '外資及陸資投資持股統計': ["外資及陸資尚可投資股數","全體外資及陸資持有股數","外資及陸資共用法令投資上限比率","陸資法令投資上限比率","發行股數","外資及陸資尚可投資比率","全體外資及陸資持股比率", '外資尚可投資股數', '全體外資持有股數', '外資尚可投資比率', '全體外資持股比率', '法令投資上限比率']
    },
    '發行量加權股價指數歷史資料': {
        '發行量加權股價指數歷史資料': ['開盤指數', '最高指數', '最低指數', '收盤指數']
    },
}

datecol = {
    'stocklist': ['date', '發行日', '到期日', '上市日', '掛牌日', '公開發行日', '登錄日', '發布日']
}

#因為漲跌的欄位用不到也很難用，所以直接drop掉
dropcol = ['漲跌(+/-)']

#這裡是全部爬蟲會用到的相關資訊
#date min是這個指數的起始時間
collection = {
    "每日收盤行情": {
        'url': r'https://www.twse.com.tw/exchangeReport/MI_INDEX?',
        'payload': {
            'response': 'json',
            'date': '',
            'type': 'ALL',
            '_': '1613296592078'
            },
        'freq': 'D',
        'datemin': '2004-2-11',
        # 'nomatch': ["大盤統計資訊", "漲跌證券數合計"],
        'subtitle': ["價格指數(臺灣證券交易所)", "價格指數(跨市場)", "價格指數(臺灣指數公司)", "報酬指數(臺灣證券交易所)", "報酬指數(跨市場)",
                     "報酬指數(臺灣指數公司)", "大盤統計資訊", "每日收盤行情"],
    },
    "信用交易統計": {
        'url': r'https://www.twse.com.tw/exchangeReport/MI_MARGN?',
        'payload': {
            'response': 'json',
            'date': '',
            'selectType': 'ALL'
            },
        'freq': 'D',
        'datemin': '2001-1-1',
        # 'nomatch': ["信用交易統計"],
        'subtitle': ["融資融券彙總", "信用交易統計"],
        },
    "市場成交資訊": {
        'url': r'https://www.twse.com.tw/exchangeReport/FMTQIK?',
        'payload': {
            'response': 'json',
            'date': '',
            '_': '1613392395864'
            },
        'freq': 'ME',
        'datemin': '1990-1-4',
        # 'nomatch': [],
        'subtitle': ['市場成交資訊'],
        },
    "三大法人買賣金額統計表": {
        'url': r'https://www.twse.com.tw/fund/BFI82U?',
        'payload': {
            'response': 'json',
            'dayDate': '',
            'type': 'day',
            '_': '1613389589646'
            },
        'freq': 'D',
        'datemin': '2004-4-7',
        # 'nomatch': ['三大法人買賣金額統計表'],
        'subtitle': ['三大法人買賣金額統計表'],
        },
    "三大法人買賣超日報": {
        'url': r'https://www.twse.com.tw/fund/T86?',
        'payload': {
            'response': 'json',
            'date': '',
            'selectType': 'ALL'
            },
        'freq': 'D',
        'datemin': '2012-5-2',
        # 'nomatch': [],
        'subtitle': ["三大法人買賣超日報"],
        },
    "個股日本益比、殖利率及股價淨值比": {
        'url': r'https://www.twse.com.tw/exchangeReport/BWIBBU_d?',
        'payload': {
            'response': 'json',
            'date': '',
            'selectType': 'ALL',
            '_': '1596117278906'
            },
        'freq': 'D',
        'datemin': '2012-5-2',
        # 'nomatch': [],
        'subtitle': ['個股日本益比、殖利率及股價淨值比'],
        },
    "信用額度總量管制餘額表": {
        'url': r'https://www.twse.com.tw/exchangeReport/TWT93U?',
        'payload': {
            'response': 'json',
            'date': '',
            '_': '1596721575815'
            },
        'freq': 'D',
        'datemin': '2005-7-1',
        # 'nomatch': ['信用額度總量管制餘額表'],
        'subtitle': ['信用額度總量管制餘額表'],
        },
    "當日沖銷交易標的及成交量值": {
        'url': r'https://www.twse.com.tw/exchangeReport/TWTB4U?',
        'payload': {
            'response': 'json',
            'date': '',
            'selectType': 'All',
            '_': '1596117305431'
            },
        'freq': 'D',
        'datemin': '2014-1-6',
        # 'nomatch': ['當日沖銷交易統計資訊'],
        'subtitle': ['當日沖銷交易統計資訊', '當日沖銷交易標的及成交量值'],
        },
    # 這裡的,'當日沖銷交易統計'跟market有重複，因為都是大盤的沖銷交易===========
    "每月當日沖銷交易標的及統計": {
        'url': 'https://www.twse.com.tw/exchangeReport/TWTB4U2?',
        'payload': {
            'response': 'json',
            'date': '',
            'stockNo': '',
            '_': '1596117360962'
            },
        'freq': 'ME',
        'datemin': '2014-1-6',
        # 'nomatch': ['每月當日沖銷交易標的及統計'],
        'subtitle': ['每月當日沖銷交易標的及統計'],
        },
    "外資及陸資投資持股統計": {
        'url': 'https://www.twse.com.tw/fund/MI_QFIIS?',
        'payload': {
            'response': 'json',
            'date': '',
            'selectType': 'ALLBUT0999',
            '_': '1594606204191'
            },
        'freq': 'D',
        'datemin': '2004-2-11',
        # 'nomatch': [],
        'subtitle': ['外資及陸資投資持股統計'],
        },
    "發行量加權股價指數歷史資料": {
        'url': 'https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?',
        'payload': {
            'response': 'json',
            'date': '',
            '_': '1597539490294'
            },
        'freq': 'D',
        'datemin': '1999-1-5',
        # 'nomatch': ['發行量加權股價指數歷史資料'],
        'subtitle': ['發行量加權股價指數歷史資料'],
        },
    }

dailycollection = {
    'stocklist': {
        'url': r'https://isin.twse.com.tw/isin/C_public.jsp?strMode={}',
        'modelis': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }
}

