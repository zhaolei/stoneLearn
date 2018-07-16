
#import MySQLdb
import pymysql
import time
import datetime
import quandl
import pandas as pd
import json
import urllib.request
import requests

quandl.ApiConfig.api_key = "GRFAUD2HY43XdKsfj8Az"

wd = datetime.date.today().weekday()

def getld():
    dd = datetime.date.today() - datetime.timedelta(days=1)
    while dd.weekday() == 6 or dd.weekday() == 5:
        dd = dd - datetime.timedelta(days=1)

    return dd

    #Hos = quandl.get("WIKI/%s"%cc,start_date = st, end_date = ed)
    #print(Hos)
    #return Hos

# 打开数据库连接
db = pymysql.connect("localhost","root","root","stone" )
 
# 使用cursor()方法获取操作游标 
cursor = db.cursor()
 

allx = 'QQQ'
#allx = 'AAPL'
alls = allx.split(',')

nnw = getld()
nw = nnw.strftime('%Y%m%d')

#nw = time.strftime("%Y%m%d")
st = '20120101'
# SQL 查询语句
sql = "SELECT name, max(datex) FROM stock where name='%s'" 
isql = "INSERT INTO `stock` (`name`, `datex`, `open`, `high`, `low`, `close`, `num`) VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s');"

for bb in alls: 
    try:
        # 执行SQL语句
        '''
        dsql = sql%bb
        cursor.execute(dsql)
        db.commit()
        # 获取所有记录列表
        results = cursor.fetchall()
        print(results)
        gnw = results[0][1]
         
        #print(results)
        #Hos = quandl.get("WIKI/%s"%bb,start_date = gnw, end_date = nw)
        if gnw is None:
            gnw = '20120101'

        '''
        url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=%s&apikey=10VRJP4Q0FNDKQ8G"%bb
        '''
        f = urllib.request.urlopen(url)
        strc = f.read()
        data = json.loads(strc)
        '''
        dat = requests.get(url)
        data = dat.json()
        
        dlist = data["Time Series (Daily)"]; 
        for dr in dlist:
            da = dlist[dr]
            tx01 = dr.replace('-','') 
            dv = list(da.values())
            ydv=isql%(bb,tx01, dv[0], dv[1], dv[2],dv[3], dv[4])
            print(ydv)
            cursor.execute(ydv)
            db.commit()
            print(bb, tx01)
        
        
           
    except:
        print ("Error: unable to fetch data***")
# 关闭数据库连接
db.close()

