import pytz
import time
import datetime
tz = pytz.timezone('America/New_York')
a = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
print(a)
a = datetime.datetime.now(tz) + datetime.timedelta(days=1)
print(a)
a = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(a)
print(datetime.date.today())
dd = datetime.date.today() - datetime.timedelta(days=1)
print(dd)
#dd = datetime.datetime.now(tz) - datetime.timedelta(days=1)
print(dd)
