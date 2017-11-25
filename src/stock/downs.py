import quandl
import pandas as pd
#quandl.ApiConfig.api_key = "GRFAUD2HY43XdKsfj8Az"
#Hos = quandl.get("WIKI/HOS",start_date = '2010-01-01', end_date = '2017-04-14')

#listk = ['BIDU','GPRO','WB','NVDA','KO','ATVI','JD','DD','MKC','MU','NKE','INTC','ORCL','AMZN','MSFT','SNE','EA','AAPL','SBUX','IBKR','NFLX','WMT','ADBE']
#listk = ['BIDU','NVDA','KO','ATVI','DD','MKC','MU','NKE','INTC','ORCL','AMZN','MSFT','EA','AAPL','SBUX','IBKR','NFLX','WMT','ADBE']
listk = ['JD']
for kk in listk:
    Hos = quandl.get("WIKI/%s"%kk,start_date = '2015-01-01', end_date = '2017-05-16')
    ffile = '/ds/datas/stock/%s'%kk
    Hos.to_pickle(ffile)
    print( "wiki %s "%kk)
    

