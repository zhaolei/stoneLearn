# -*- coding: utf-8 -*-
import pymysql
import time
import datetime
import quandl
import pandas as pd
import base

quandl.ApiConfig.api_key = "GRFAUD2HY43XdKsfj8Az"

wd = datetime.date.today().weekday()

def getld():
    dd = datetime.date.today() - datetime.timedelta(days=1)
    while dd.weekday() == 6 or dd.weekday() == 5:
        dd = dd - datetime.timedelta(days=1)

    return dd


'''
# 打开数据库连接
db = pymysql.connect("localhost","root","root","stone" )
# 使用cursor()方法获取操作游标 
cursor = db.cursor()
nsql = 'select name from stock group by name' 
cursor.execute(nsql)
db.commit()
results = cursor.fetchall()
alls = [x[0] for x in results]
'''

alls = base.getList()

#allx = 'AAPL,ADBE,AMZN,ATVI,BIDU,EA,IBKR,INTC,KO,MKC,MSFT,MU,NFLX,NKE,NVDA,ORCL,SBUX,WMT,FB'
#allx = 'AAPL'
#alls = allx.split(',')

nnw = getld()
nw = nnw.strftime('%Y%m%d')

#nw = time.strftime("%Y%m%d")
st = '20120101'
# SQL 查询语句
sql = "SELECT name, max(datex) FROM stock where name='%s'" 
isql = "INSERT INTO `stock` (`name`, `datex`, `open`, `high`, `low`, `close`, `num`) VALUES ('%s', '%s', '%f', '%f', '%f', '%f', '%d');"

for bb in alls: 
    try:
        # 执行SQL语句
        dsql = sql%bb
        base.cursor.execute(dsql)
        base.db.commit()
        # 获取所有记录列表
        results = base.cursor.fetchall()
        print(results)
        gnw = results[0][1]
         
        #print(results)
        #Hos = quandl.get("WIKI/%s"%bb,start_date = gnw, end_date = nw)
        if gnw is None:
            gnw = '20120101'

        Hos = quandl.get("WIKI/%s"%bb,start_date = str(gnw), end_date = str(nw))
        print(Hos.values.shape)
        print(Hos.index)
        for di,dv in zip(Hos.index[1:], Hos.values[1:]):
            tx01 = di.strftime('%Y%m%d')
            ydv=isql%(bb,tx01, dv[0], dv[1], dv[2],dv[3], dv[4])
            base.cursor.execute(ydv)
            base.db.commit()
            print(bb, tx01)
        
        
    except:
        print ("Error: unable to fetch data***")
# 关闭数据库连接
base.db.close()

print(nw)
