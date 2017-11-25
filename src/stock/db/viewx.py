import slist
import pymysql
import keras
from keras.models import load_model

db = pymysql.connect("localhost","root","root","stone" )
cursor = db.cursor()

alls = slist.listcc

def getM(cc):
    n = 0
    model = load_model('/ds/model/stock/ls_%s_%d.h5'%(cc,n))
    return model

for cc in alls:
    dsql = "SELECT * FROM `stock_predict` where name='%s' order by datex asc limit 5"
    cursor.execute(dsql%'aa')
    results = cursor.fetchall()
    lx = list(results)
    print('<hr />')
    print('code : %s <br />'%cc)
    for t01 in re
    print(lx)
    
