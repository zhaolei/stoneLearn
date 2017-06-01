import quandl
import pandas as pd

def get_data(codex):
    quandl.ApiConfig.api_key = "GRFAUD2HY43XdKsfj8Az"
    #Hos = quandl.get("WIKI/HOS",start_date = '2010-01-01', end_date = '2017-04-14')
    Hos = quandl.get("WIKI/%s"%codex,start_date = '2014-01-01', end_date = '2017-04-14')
    return Hos

def get_local_data(codex):
    
    ffile = '/ds/datas/stock/%s'%codex
    da = pd.read_pickle(ffile)
    return da

def unify_data(datax):
    for i in range(datax.shape[1]):
        sumx = np.sqrt(sum(pow(datax[:,i])))
        datax[:,i] /= (sumx * 1.0)

    return datax
