import json
import urllib.request

isql = "INSERT INTO `stock` (`name`, `datex`, `open`, `high`, `low`, `close`, `num`) VALUES ('%s', '%s', '%f', '%f', '%f', '%f', '%d');"

bb = 'BIDU'
url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&apikey=10VRJP4Q0FNDKQ8G"%bb
f = urllib.request.urlopen(url)
strc = f.read()
data = json.loads(strc)

dlist = data["Time Series (Daily)"];
for dr in dlist:
    da = dlist[dr]
    tx01 = dr.replace('-','')
    dv = list(da.values())
    ydv=isql%(bb,tx01, dv[0], dv[1], dv[2],dv[3], dv[4])
    print(ydv)
    exit()

