

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
                     "報酬指數(臺灣指數公司)", "大盤統計資訊", "每日收盤行情","漲跌證券數合計"],
        "code" : "A"
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
        "code" : "B"
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
        "code" : "C"
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
        "code" : "D"
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
        "code" : "E"
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
        "code" : "F"
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
        "code" : "G"
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
        "code" : "H"
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
        "code" : "I"
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
        "code" : "J"
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
        "code" : "K"
        },
    }

dailycollection = {
    'stocklist': {
        'url': r'https://isin.twse.com.tw/isin/C_public.jsp?strMode={}',
        'modelis': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }
}

