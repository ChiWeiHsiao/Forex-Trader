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
STD = 0.00003
STD = 0.00006
#STD = 0.0005
ORDER = 20000

TAKE_PROFIT = 0.00015
STOP_LOSS = 0.00015

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

        if std < STD:
            priceLine = np.mean(cArr)
        else:
            priceLine = -1
        #print "cur=", cur, ", std=", std
        #print "priceLine=", priceLine
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
    times = 0
    period = 1200
    print "====System Start===="
    print "Use STD: ", STD
    print "Use period: ", period
    start_balance = getAccountBalance()

    # times = 1 hour / 3 = 60 * 60 / 3
    while times < period:
        times = times + 1
        fromTime = (datetime.utcnow() - timedelta(hours=0, minutes=2, seconds=0)).isoformat()+"Z"
        toTime = datetime.utcnow().isoformat()+"Z"
        action =  getCandle(fromTime, toTime, 'M', 'S5')
        '''
        action == -1 -> fail to get candles
        action == 0 -> stall
        action == 1 -> buy
        action == 2 -> sell
        '''
        if action != 0 and action != -1:
            print "\nRun#", times
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

    end_balance = getAccountBalance()
    print "====System End===="
    print "Earn money: ", float(end_balance) - float(start_balance)
