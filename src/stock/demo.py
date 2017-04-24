import quandl
Hos = quandl.get("WIKI/HOS",start_date = '2010-01-01', end_date = '2017-04-14')

print Hos.head(5)
