import matplotlib.pyplot as plt
import json
import numpy as np

filename = 'statistics/H6_%s_rnn_MA_return_%s_train.json' %('smallData-RMS1e7-Dense-2LSTM20-noDrop', 'ep3')

with open(filename, 'r') as f:
  log = json.load(f)
  real_price = log['real_price']
  predict_price = log['predict_price']

def plot_curve(name, statistics_1, statistics_2):
  show_low = True
  show_high = False

  win = 3
  high_x1, high_y1, low_x1, low_y1 = [], [], [], []
  high_x2, high_y2, low_x2, low_y2 = [], [], [], []
  max_s1 = max(statistics_1)
  min_s1 = min(statistics_1)
  for i in range(len(statistics_1)):
    print(i, end='')
    if i < win:
      continue
    if statistics_1[i] == min(statistics_1[i-win:i+win]):
      low_x1 += [i, i]
      low_y1 += [min_s1, max_s1]
    if statistics_1[i] == max(statistics_1[i-win:i+win]):
      high_x1 += [i, i]
      high_y1 += [min_s1, max_s1]
    if statistics_2[i] == min(statistics_2[i-win:i+win]):
      low_x2 += [i, i]
      low_y2 += [min_s1, max_s1]
    if statistics_2[i] == max(statistics_2[i-win:i+win]):
      high_x2 += [i, i]
      high_y2 += [min_s1, max_s1]
  plt.figure(name)
  if show_high:
    print('Nun of high peak = ', len(high_x1)/2)
    for i in range(int(len(high_x1)/2)):
      plt.plot(high_x1[2*i:2*i+2], high_y1[2*i:2*i+2], 'b--', alpha=0.5)
    for i in range(int(len(high_x2)/2)):
      plt.plot(high_x2[2*i:2*i+2], high_y2[2*i:2*i+2], 'r--', alpha=0.5)

  if show_low:
    for i in range(int(len(low_x1)/2)):
      plt.plot(low_x1[2*i:2*i+2], low_y1[2*i:2*i+2], 'b--', alpha=0.5)
    
    for i in range(int(len(low_x2)/2)):
      plt.plot(low_x2[2*i:2*i+2], low_y2[2*i:2*i+2], 'r--', alpha=0.5)

  plt.plot(statistics_1, 'b', label='real')
  plt.plot(statistics_2, 'r', label='predict')
  plt.xlabel('Time')
  plt.ylabel('Close price')
  plt.legend()
  plt.show()
  #plt.savefig('images/'+id+'_'+typ[:4]+'.png')


if __name__ == '__main__':
  plot_curve('ep_show_', real_price, predict_price)

