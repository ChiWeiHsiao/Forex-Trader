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

fp = open('history.txt', 'w')


def saveMany():
		fromTime = datetime.utcnow() - timedelta(days = 60)
		#fp.write('from time: ')
		#fp.write(fromTime.strftime('%Y-%m-%d %H:%M:%S'))
		#fp.write('\n')
		for i in range(1):
			save(fromTime)
			fromTime = fromTime + timedelta(days=1, hours=17, minutes = 40) #=2500min

def save(fromT):
	fromTime = fromT.isoformat()+'Z'
	count = '5000'
	granularity = 'S15'
	instr = 'EUR_USD'

	url = 'https://api-fxpractice.oanda.com/v3/instruments/'+instr+'/candles'
	query = {'from': fromTime, 'count': count, 'price': 'M', 'granularity': granularity }
	r = requests.get(url, headers=headers, params=query)
	if r.status_code != 200:
		print('Save History Candle, Fail: ', r.status_code)
		return -1
	else:
	candles = json.loads(r.text)['candles']

	#overlap
	for start in range(4951):
	for j in range(49):
		diff = float(candles[start+j+1]['mid']['c']) - float(candles[start+j]['mid']['c'])
		fp.write(str(diff))
		fp.write(' ')
	fp.write('\n')

	""" 
	#non-overlap
	for i in range(100):
		for j in range(49):
			diff = float(candles[j+1]['mid']['c']) - float(candles[j]['mid']['c'])
			fp.write(str(diff))
			fp.write(' ')
		fp.write('\n')
	"""				 

if __name__ == '__main__':
	saveMany()
	fp.close()
