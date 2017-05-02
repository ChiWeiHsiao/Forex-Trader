'''
save one year candles [c,h,l,o] data to .npz file
(day, 10_min, candle) = (46*4, 144, 4)
'''
from datetime import datetime, timedelta
import numpy as np
import time 

X = []
Y = []
year = []

def export():
  X = np.array(X) 
  Y = np.array(Y) 
  np.savez(filename, X=X, Y=Y)


if __name__ == '__main__':
  candles = np.load('data/candles_05-17.npy')
  print(candles.shape) #(2268, 144, 4)  
  # candle = (c, h, l, o)
  
  ### Extract features ###
  ''' log returns
  for i in range(143):
    if i == 0:
      log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
      log_return = np.reshape(log_return, (-1,1))
      print(log_return.shape)
    else:
      new_log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
      new_log_return = np.reshape(new_log_return, (-1,1))
      log_return = np.append(log_return, new_log_return, axis=1)
  print(log_return.shape) #(2268, 143)
  np.save('data/log_return', log_return)
  '''
  # 
  for i in range(143):
    if i == 0:
      log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
      log_return = np.reshape(log_return, (-1,1))
      print(log_return.shape)
    else:
      new_log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
      new_log_return = np.reshape(new_log_return, (-1,1))
      log_return = np.append(log_return, new_log_return, axis=1)
  print(log_return.shape) #(2268, 143)
  np.save('data/log_return', log_return)


  # Split to Training set (2000) and Testing set (268)
