import matplotlib.pyplot as plt
import json
import numpy as np

filename = 'statistics/H6_LSTM_rnn_features_ep300_train.json'


with open(filename, 'r') as f:
  log = json.load(f)
  real_price = log['real_price']
  predict_price = log['predict_price']

def plot_curve(name, statistics_1, statistics_2):
  plt.figure(name)
  plt.plot(statistics_1, label='real')
  plt.plot(statistics_2, label='predict')
  plt.xlabel('Time')
  plt.ylabel('Close price')
  plt.legend()
  plt.show()
  #plt.savefig('images/'+id+'_'+typ[:4]+'.png')


if __name__ == '__main__':
  plot_curve('Close Price', real_price, predict_price)

