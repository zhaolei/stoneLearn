'''
import sqlite3
dpath = '/ds/datas/stock/nsdq.db'
conn = sqlite3.connect(dpath)
print(conn)
'''
import pandas as pd
import time


#'Open', 'High', 'Low', 'Close', 'Volume'
allx = 'AAPL,ADBE,AMZN,ATVI,BIDU,DD,EA,IBKR,INTC,KO,MKC,MSFT,MU,NFLX,NKE,NVDA,ORCL,SBUX,WMT'
alls = allx.split(',')

def tosql(dx):
    print(1)


sql = "INSERT INTO `stock` (`name`, `datex`, `open`, `high`, `low`, `close`, `num`) VALUES ('%s', '%s', '%f', '%f', '%f', '%f', '%d');"
for bb in alls:
    ppx = '/ds/datas/stock/%s'%bb
    fh = pd.read_pickle(ppx)
    print(fh.columns)
    for di,dv in zip(fh.index, fh.values):
        tx01 = di.strftime('%Y%m%d')
        
        #ydv='%s\t%s\t%f\t%f\t%f\t%f\t%d'%(bb,tx01, dv[0], dv[1], dv[2],dv[3], dv[4])
        ydv=sql%(bb,tx01, dv[0], dv[1], dv[2],dv[3], dv[4])
        print(ydv)


