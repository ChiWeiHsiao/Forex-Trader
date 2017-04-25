import ConfigParser
import requests
import json
from datetime import datetime, timedelta
import time 

config = ConfigParser.ConfigParser()
config.read('../api.config')
OANDA_ACCESS_TOKEN = config.get('account', 'token')
OANDA_ACCOUNT_ID = config.get('account', 'accountID')
headers = {"Content-Type": 'application/json' ,'Authorization': 'Bearer '+OANDA_ACCESS_TOKEN}

fp = open("history.txt", "w")

def saveHistoryNodes():
    fromTime = datetime.utcnow() - timedelta(days = 60)
    fp.write("from time: ")
    fp.write(fromTime.strftime("%Y-%m-%d %H:%M:%S"))
    fp.write("\n")

    for i in range(10):
        saveOneHundredNodes(fromTime)
        fromTime = fromTime + timedelta(days=1, hours=17, minutes = 40) #=2500min

def saveOneHundredNodes(fromT):
    fromTime = fromT.isoformat()+"Z"
    count = '5000'
    granularity = 'S15'
    instr = "EUR_USD"

    url = "https://api-fxpractice.oanda.com/v3/instruments/"+instr+"/candles"
    query = {'from': fromTime, 'count': count, 'price': 'M', 'granularity': granularity }
    r = requests.get(url, headers=headers, params=query)
    if r.status_code != 200:
        print "Save History Candle, Fail: ", r.status_code
        return -1
    else:
        candles = json.loads(r.text)['candles']
        for candle in candles:
            fp.write(candle['mid']['c'])
            fp.write(",")
        fp.write("\n")

if __name__ == "__main__":
    saveHistoryNodes()
    fp.close()
