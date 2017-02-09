import ConfigParser
import requests
import json
from datetime import datetime, timedelta
import numpy as np
import time 

config = ConfigParser.ConfigParser()
config.read('api.config')
OANDA_ACCESS_TOKEN = config.get('account', 'token')
OANDA_ACCOUNT_ID = config.get('account', 'accountID')
headers = {"Content-Type": 'application/json' ,'Authorization': 'Bearer '+OANDA_ACCESS_TOKEN}

priceLine = -1
PRICELINE_OFFSET = 0.00001
ORDER = 20000

TAKE_PROFIT = 0.00015
STOP_LOSS = 0.00015

#parameters
instr = "EUR_USD"
base_units = 1000
history = []
kval = 5

def saveHistoryNodes():
    # 50 candle prices in one node (25min), 100 nodes in history(1.7day)
    fromTime = (datetime.utcnow() - timedelta(days=2, hours=0, minutes=0, seconds=0)).isoformat()+"Z"
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
        index = 0
        fp = open("history.txt", "w")

        for i in range(100):
            node = []
            for j in range(50):
                node.append(float(candles[index]['mid']['c']))
                index = index + 1
            history.append(node)
        
        fp.close()

def getCurNode():
    toTime = datetime.utcnow().isoformat()+"Z"
    count = '50'
    granularity = 'S15'
    url = "https://api-fxpractice.oanda.com/v3/instruments/"+instr+"/candles"
    query = {'count': count, 'to': toTime, 'price': 'M', 'granularity': granularity}
    r = requests.get(url, headers=headers, params=query)
    if r.status_code != 200:
        print "getCurNode, Fail: ", r.status_code
        return -1
    else:
        candles = json.loads(r.text)['candles']
        cur = float (candles[-1]['mid']['c'])
        node = []
        for i in range(50):
            node.append( float(candles[i]['mid']['c']) )

def predictChange():
    # use KNN-like mmethod to predict if the price will rise/drop
    # [0] is oldest, [-1] is newest

    # a list with kval(5) tuples
    closestNodes = [] 
    for i in range(kval):
        closestNodes.append((1000.0,1000.0))

    # calculate the distance between history-nodes and current-node
    # keep the kval(5) closest nodes in the closestNodes[]
    for node in history:
        print

    
def decideOrderUnits():
    units = 0
    print "decide to order #", units, "units"

def getCandle(fromTime, toTime, priceType, granularity):
    instr = "EUR_USD"
    url = "https://api-fxpractice.oanda.com/v3/instruments/"+instr+"/candles"
    query = {'from': fromTime, 'to': toTime, 'price': priceType, 'granularity': granularity}
    r = requests.get(url, headers=headers, params=query)
    if r.status_code != 200:
        print "getCandle, Fail: ", r.status_code
        return -1
    else:
        candles = json.loads(r.text)['candles']
        cur = float (candles[-1]['mid']['c'])
        cArr = []
        for i in range(0, len(candles)-2):
            cArr.append( float(candles[i]['mid']['c'] ))
        cArr = np.array(cArr)
        std = np.std( cArr, dtype=np.float32)

        print "candle 0: ", candles[0]['time']
        print "candle -1: ", candles[-1]['time']
        return -1

        if std < STD:
            priceLine = np.mean(cArr)
        else:
            priceLine = -1
        print "cur=", cur, ", std=", std
        print "priceLine=", priceLine

        if priceLine==-1 or np.absolute(cur - priceLine) < PRICELINE_OFFSET:
            return 0 #stall
        elif cur - priceLine > 0:
            return 1 #buy
        else:
            return 2 #sell

def buySell(buyOrSell):
    url = "https://api-fxpractice.oanda.com/v3/accounts/"+OANDA_ACCOUNT_ID+"/orders"
    if buyOrSell == 1:
        units = str(ORDER)
    else:
        units = '-' + str(ORDER)

    body={
        'order': {
            'type' : 'MARKET',
            'instrument' : 'EUR_USD',
            'units' : units,
            'timeInForce' : 'FOK',  #Filled Or Killed
            'positionFill': "DEFAULT"
        }
    }
    r = requests.post( url, headers=headers, data=json.dumps(body) )
    data = r.json()
    if r.status_code == 201:
        if buyOrSell == 1:
            print "Buy, Success"
        else:
            print "Sell, Success"
    else:
        print "BuySell, Fail: ", data['errorMessage']#r.status_code
        return (-1,-1)
    orderID = data['orderFillTransaction']['id'] #data['lastTransactionID']
    price = data['orderFillTransaction']['price']
    print "buy id: ", orderID
    return ( int(orderID), float(price) )

def takeProfit(tradeID, threshold):
    #threshold = 1.0600
    print "take id: ", tradeID
    url = "https://api-fxpractice.oanda.com/v3/accounts/"+OANDA_ACCOUNT_ID+"/orders"
    body={
        'order': {
            'type': 'TAKE_PROFIT',
            'timeInForce': 'GTC', #Good unTil Cancelled
            'price': str(threshold),
            'tradeID': str(tradeID) #'48'
        }
    }
    r = requests.post( url, headers=headers, data=json.dumps(body) )
    if r.status_code == 201:
        print "Take Profit Order, Success"
    else:
        print "Take Profit Order, Fail: ", r.status_code, ": ", r.json()['errorMessage']

def stopLoss(tradeID, threshold):
    #threshold = 0.9600
    url = "https://api-fxpractice.oanda.com/v3/accounts/"+OANDA_ACCOUNT_ID+"/orders"
    body={
        'order': {
            'type': 'STOP_LOSS',
            'timeInForce': 'GTC', #Good unTil Cancelled
            'price': str(threshold),
            'tradeID': str(tradeID)
        }
    }
    r = requests.post( url, headers=headers, data=json.dumps(body) )
    if r.status_code == 201:
        print "Stop Loss Order, Success"
    else:
        print "Stop Loss Order, Fail: ", r.status_code, ": ", r.json()['errorMessage']

def getAccountBalance():
    url = "https://api-fxpractice.oanda.com/v3/accounts/"+OANDA_ACCOUNT_ID
    r = requests.get( url, headers=headers )
    if r.status_code == 200:
        cur_balance = r.json()['account']['balance']
        print "Balance = ", cur_balance
        return cur_balance
    else:
        print "GetAccountBalance, Fail: ", r.json()['errorCode'], ": ", r.json()['errorMessage']
        return -1

if __name__ == "__main__":
    saveHistoryNodes()
    #getCurNode()
    




def run():
    times = 0
    print "====System Start===="
    start_balance = getAccountBalance()

    # times = 1 hour / 3 = 60 * 60 / 3
    while times < 1200:
        times = times + 1
        print "Run#", times
        fromTime = (datetime.utcnow() - timedelta(hours=0, minutes=0, seconds=100)).isoformat()+"Z"
        toTime = (datetime.utcnow() - timedelta(hours=0, minutes=0, seconds=20)).isoformat()+"Z"
        action =  getCandle(fromTime, toTime, 'M', 'S5')
        '''
        action == -1 -> fail to get candles
        action == 0 -> stall
        action == 1 -> buy
        action == 2 -> sell
        '''
        if action != 0 and action != -1:
            buySell_return = buySell(action)
            orderID = buySell_return[0]
            price = buySell_return[1]
            if orderID != -1: #buySell() success
                print "From: ", fromTime
                print "To: ", toTime
                print "buy/sell at price: ",price
                #calculate the thresholds for stopLoss(), takeProfit
                if(action == 1):
                    stopPrice = price - STOP_LOSS
                    takePrice = price + TAKE_PROFIT
                else:
                    stopPrice = price + STOP_LOSS
                    takePrice = price - TAKE_PROFIT
                stopLoss(orderID, stopPrice)
                takeProfit(orderID, takePrice)
        time.sleep(2)
        print

    end_balance = getAccountBalance()
    print "====System End===="
    print "Earn money: ", float(end_balance) - float(start_balance)
