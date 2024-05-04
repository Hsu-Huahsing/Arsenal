import pandas as pd
from StevenTricks.fileop import pickleload, picklesave, warehouseinit
from StevenTricks.dfi import periodictable
from os.path import exists, join
from datetime import datetime, timedelta
datetime.now()


class Log:
    def __init__(self, warehousepath=''):
        self.warehousepath = warehousepath
        warehouseinit(self.warehousepath)

    def findlog(self, logtype, kind):
        # logtype could be 'source' 'cleaned' ''
        # kind could be 'log.pkl' 'errorlog.pkl'
        # print(join(self.warehousepath, logtype, kind))
        if exists(join(self.warehousepath, logtype, kind)) is True:
            print('exists')
            return pickleload(join(self.warehousepath, logtype, kind))
        return None

    def updatelog(self, periodictdf, periodict):
        if periodictdf is None:
            log = periodictable(periodict)
        else:
            if str(datetime.today().date()) not in periodictdf.index:
                latestlog = periodictable(periodict, datemin=periodictdf.index.max()+timedelta(days=1))
                log = pd.concat([periodictdf, latestlog])
            else:
                log = periodictdf
        return log

    def savelog(self, log, logtype, kind):
        # logtype could be 'source'、'cleaned'，也可以什麼都不打 '' ，就代表是warehouse底下的使用紀錄
        # kind could be 'log.pkl' 'errorlog.pkl'
        path = join(self.warehousepath, logtype, kind)
        picklesave(log, path)


if __name__ == '__main__':
    pass
