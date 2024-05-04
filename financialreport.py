#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 21:34:14 2021

@author: stevenhsu
"""

import requests
from bs4 import BeautifulSoup as bf
import pandas as pd

# b = requests.get(r"https://mops.twse.com.tw/server-java/t164sb01?step=3&SYEAR=2020&file_name=tifrs-fr1-m1-ci-cr-3706-2020Q3.html#BalanceSheet")

a = pd.read_html(r"https://mops.twse.com.tw/server-java/t164sb01?step=3&SYEAR=2020&file_name=tifrs-fr1-m1-ci-cr-3706-2020Q3.html#BalanceSheet")
