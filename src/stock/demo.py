import quandl
import pandas as pd
import json
import urllib.request

quandl.ApiConfig.api_key = "GRFAUD2HY43XdKsfj8Az"
#Hos = quandl.get("WIKI/HOS",start_date = '2010-01-01', end_date = '2017-04-14')
#Hos = quandl.get("WIKI/BIDU",start_date = '2012-01-01', end_date = '2017-09-20')
#Hos = quandl.get("WIKI/BIDU",start_date = '20171105', end_date = '20171122')
#Hos = quandl.get_table('WIKI/PRICES', qopts = { 'columns':['ticker', 'date', 'close'] }, ticker = ['AAPL', 'MSFT'], date = { 'gte': '2017-11-06', 'lte': '2017-11-09' })


#print( Hos.values.shape)
#print(Hos.head(5))
#print(Hos.index)
#print(type(Hos.index[0]))
#print(Hos.values.shape)
#print(Hos)
'''
print dir(Hos)
print Hos.columns
#print Hos.Close
print Hos.values[10][0]
Hos.to_pickle('bidu.d')
fh = pd.read_pickle('bidu.d')
print fh.values.shape
print fh.columns
'''
url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=10VRJP4Q0FNDKQ8G"
f = urllib.request.urlopen(url)
strc = f.read()
data = json.loads(strc)
print(data)


