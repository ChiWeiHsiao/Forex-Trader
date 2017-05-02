'''
save one year candles [c,h,l,o] data to .npz file
(day, 10_min, candle) = (46*4, 144, 4)
'''
import configparser
import requests
import json
from datetime import datetime, timedelta
import numpy as np
import time 

config = configparser.ConfigParser()
config.read('../api.config')
OANDA_ACCESS_TOKEN = config.get('account', 'token')
OANDA_ACCOUNT_ID = config.get('account', 'accountID')
headers = {'Content-Type': 'application/json' ,'Authorization': 'Bearer '+OANDA_ACCESS_TOKEN}
year = []

def findFirstMonday(dt):
  while dt.isoweekday() != 1:
    dt = dt + timedelta(days=1)
  return dt


def saveOneYear(which_year):
  fromTime = which_year
  # Save 46 weeks of 1 year
  for i in range(15):
    # Save 4 days of 1 week
    #print('From time, week %d: %s' %(i, fromTime.strftime('%A, %Y-%m-%d %H:%M:%S\t')))
    for i in range(4):
      save(fromTime+timedelta(days=i), count=144)
      #save(fromTime+timedelta(days=i), fromTime+timedelta(days=i+1))
    fromTime = fromTime + timedelta(weeks=1) 


def save(fromT, count):
  fromTime = fromT.isoformat()+'Z'
  #toTime = toT.isoformat()+'Z'
  granularity = 'M10'
  instr = 'EUR_USD'
  url = 'https://api-fxpractice.oanda.com/v3/instruments/'+instr+'/candles'
  query = {'from': fromTime, 'count': count, 'price': 'B', 'granularity': granularity}
  r = requests.get(url, headers=headers, params=query)
  if r.status_code != 200:
    print('Save History Candle, Fail: ', r.status_code)
    return -1
  else:
    candles = json.loads(r.text)['candles']
    
  day = []
  for c in candles:
    day.append([ float(c['bid']['c']), float(c['bid']['h']), float(c['bid']['l']), float(c['bid']['o']) ])
    #print(str(c['time']))
  year.append(day)
  #print('Last candle is: ', candles[-1]['time'], '\n')
  #print('total length: ', len(candles))
  return 


if __name__ == '__main__':
  # Select year, [2005,...,2016]
  for y in range(2017,2018):
    year = []
    print('%d year processing...' %(y) )
    utc_time = findFirstMonday(datetime(y, 1, 3, 0, 0, 00)) #+ timedelta(hours=5))
    filename = 'data/raw_candles_%d' %(y)
    saveOneYear(utc_time)
    print('nparray shape:')
    year = np.array(year)
    print(year.shape)
    np.save(filename, year)
