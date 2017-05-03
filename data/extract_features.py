'''
(day, 10_min, candle) = (46(weeks)*4(days), 144(10min), 4(c,h,l,o))
'''
from datetime import datetime, timedelta
import numpy as np
import time 

def extract_features():
  candles = np.load('data/candles_05-17.npy')
  # candles.shape = (2268, 144, 4) 
  # last dim is one candle: (c, h, l, o)
  # Log returns
  for i in range(143):
    if i == 0:
      log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
      log_return = np.reshape(log_return, (-1,1))
      print(log_return.shape)
    else:
      new_log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
      new_log_return = np.reshape(new_log_return, (-1,1))
      log_return = np.append(log_return, new_log_return, axis=1)
  print('shape of log_return: ', log_return.shape) #(2268, 143)
  log_return = np.reshape(log_return, (2268, 143, 1))
  #np.save('data/log_return', log_return)
  # upper_length = high_1 - max(open_3,close_0) => + 
  open_and_close = np.append(np.reshape(candles[:,:,3],(2268,144,1)), np.reshape(candles[:,:,0], (2268,144,1)), axis = 2)
  upper_length = candles[:,:,1] - np.amax(open_and_close, axis=2)
  upper_length = upper_length[:, 1:-1] # 144 -> 142
  upper_length = np.reshape(upper_length, (2268, 142, 1))
  # lower_length = min(open_3, close_0) - low => +
  open_and_close = np.append(np.reshape(candles[:,:,3],(2268,144,1)), np.reshape(candles[:,:,0], (2268,144,1)), axis = 2)
  lower_length = np.amin(open_and_close, axis=2) - candles[:,:,2]
  lower_length = lower_length[:, 1:-1]
  lower_length = np.reshape(lower_length, (2268, 142, 1))
  # whole_length = high_1 - low_2 => +
  whole_length = candles[:,:,1] - candles[:,:,2]
  whole_length = whole_length[:, 1:-1]
  whole_length = np.reshape(whole_length, (2268, 142, 1))
  # close_sub_open = close_0 - open_3 => +or-
  close_sub_open = candles[:,:,0] - candles[:,:,3]
  close_sub_open = close_sub_open[:, 1:-1]
  close_sub_open = np.reshape(close_sub_open, (2268, 142, 1))
  print('shape of close_sub_open: ', close_sub_open.shape)

  y = log_return[:,-1] # last one of log_return
  log_return = log_return[:,0:-1]
  x =  np.concatenate((log_return, upper_length, lower_length, whole_length, close_sub_open), axis=2)
  print('shape of x:', x.shape) #(2268, 142, 5)
  print('shape of y:', y.shape) #(2268, 1, 1)
  return x, y


def create_dataset(x, y):
  # Split to Training set (2000) and Testing set (268)
  x_train = x[:2000]
  y_train = y[:2000]
  x_test = x[2000:]
  y_test = y[2000:]
  print('shape of x_train, y_train:', x_train.shape, y_train.shape)
  print('shape of x_test, y_test:', x_test.shape, y_test.shape)
  np.savez('rnn_data', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def create_answer():
  candles = np.load('data/candles_05-17.npy')
  last_two = np.reshape(candles[:, -2, 0], (2268, 1))
  last_one = np.reshape(candles[:, -1, 0], (2268, 1))
  np.savez('ans_data', last_two=last_two, last_one=last_one)



if __name__ == '__main__':
  x, y = extract_features()
  create_dataset(x, y)
  create_answer()
