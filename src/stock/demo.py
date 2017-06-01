import quandl
import pandas as pd
quandl.ApiConfig.api_key = "GRFAUD2HY43XdKsfj8Az"
#Hos = quandl.get("WIKI/HOS",start_date = '2010-01-01', end_date = '2017-04-14')
Hos = quandl.get("WIKI/BIDU",start_date = '2012-01-01', end_date = '2017-04-14')

print Hos.values.shape
print Hos.head(5)
print dir(Hos)
print Hos.columns
#print Hos.Close
print Hos.values[10][0]
Hos.to_pickle('bidut')
fh = pd.read_pickle('bidut')
print fh.values.shape
print fh.columns
