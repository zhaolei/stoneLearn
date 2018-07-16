import json
import urllib.request
import requests

#lx = ['BIDU', 'JD', 'APPL', 'MU', 'TSLA', 'MSCI']
lx = ['VIX', 'VXX', '.IXIC', '.INX']

for bb in lx:
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=%s&apikey=10VRJP4Q0FNDKQ8G"%bb
    dat = requests.get(url)
    data = dat.json()

    fp = open('/ds/datas/stock/%s'%bb, 'w')
    fp.write(dat.text)
    fp.close()
        
    '''
    dlist = data["Time Series (Daily)"]; 
    for dr in dlist:
        da = dlist[dr]
        tx01 = dr.replace('-','') 
        print(tx01)
        #dv = list(da.values())
        #print(dv)
    '''
        
