import pandas as pd
from StevenTricks.dictur import findstr
from StevenTricks.fileop import pickleload, picklesave, warehouseinit
from StevenTricks.dfi import periodictable
from StevenTricks.warren.conf import collection, colname_dic
from StevenTricks.netGEN import randomheader, safereturn
import requests as re
# from sys import path
from os.path import exists, join


if __name__ == '__main__':
    pass


    def stocktablecrawl(maxn=13, timeout=180, pk="ISINCode"):
        # maxn 是指這個網頁支援的產品類型總類，目前最多到12，因此預設是13
        # dm = dbmanager(user="root")
        # dm.choosedb(db="stocktable")
        pass


