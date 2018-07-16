# -*- coding: utf-8 -*-

import pymysql

db = pymysql.connect("localhost","root","root","stone" )
cursor = db.cursor()

def getList():
    nsql = 'select name from stock group by name' 
    cursor.execute(nsql)
    db.commit()
    results = cursor.fetchall()
    alls = [x[0] for x in results]
    return alls


if __name__ == '__main__':
    s = getList()
    print(s)
