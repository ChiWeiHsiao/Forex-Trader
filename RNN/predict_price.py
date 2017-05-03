from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import json
from util import to_categorical, Dataset

architecture = 'LSTM'
eid = 'tf_' + architecture + '_3'
n_epochs = 20
batch_size = 20 #20? 4?
show_steps = 10 # show statistics after train 50 batches
learning_rate = 0.001
n_hidden = 3
shuffle = True

log = {
  'experiment_id': eid,
  'train_loss': [],
  'test_loss': [],
  'n_hidden': n_hidden,
  'batch_size': batch_size, 
  'best_loss': 1, 
  'n_epochs': n_epochs,
  'shuffle': shuffle,
}
logfile = 'statistics/'+eid+'.json'
print('id: ', eid)
print('num of epochs:', n_epochs)


#### Load data ####
data = np.load('../data/data/rnn_data.npz')
# 5 features: (log_return, upper_length, lower_length, whole_length, close_sub_open)
X_train = data['x_train'] #(2000, 142, 5)
Y_train = data['y_train'] #(2000, 1)
X_test  = data['x_test'] #(268, 142, 5)
Y_test  = data['y_test'] #(268, 1)
n_samples = X_train.shape[0]
n_timesteps = X_train.shape[1]
n_input = X_train.shape[2]
n_output = 1
n_iters = int(n_epochs * n_samples / batch_size)
print('number of iterations %d' %n_iters)
# Convert to Dataset instance 
train_dataset = Dataset(X_train, Y_train, batch_size)


def RNN(x_sequence, n_hidden):
  cell = rnn.BasicRNNCell(n_hidden)
  outputs, states = rnn.static_rnn(cell, x_sequence, dtype=tf.float32)
  # use the last output of rnn cell to compute cost function
  weight = tf.Variable(tf.random_normal([n_hidden, n_output]))
  bias = tf.Variable(tf.random_normal([n_output]))
  return tf.matmul(outputs[-1], weight) + bias
  
def LSTM(x_sequence, n_hidden):
  cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
  outputs, states = rnn.static_rnn(cell, x_sequence, dtype=tf.float32)
  # use the last output of rnn cell to compute cost function
  weight = tf.Variable(tf.random_normal([n_hidden, n_output]))
  bias = tf.Variable(tf.random_normal([n_output]))
  return tf.matmul(outputs[-1], weight) + bias

def GRU(x_sequence, n_hidden):
  cell = rnn.GRUCell(n_hidden)
  outputs, states = rnn.static_rnn(cell, x_sequence, dtype=tf.float32)
  # use the last output of rnn cell to compute cost function
  weight = tf.Variable(tf.random_normal([n_hidden, n_output]))
  bias = tf.Variable(tf.random_normal([n_output]))
  return tf.matmul(outputs[-1], weight) + bias


#### Define RNN model ####
# Graph input
x = tf.placeholder('float', [None, n_timesteps, n_input])
y = tf.placeholder('float', [None, n_output])

# Unstack to get a list of 'n_timesteps' tensors of shape (batch_size, n_input)
x_sequence = tf.unstack(x, n_timesteps, 1)
if(architecture == 'RNN'):
  predict = RNN(x_sequence, n_hidden)
elif(architecture == 'LSTM'):
  predict = LSTM(x_sequence, n_hidden)
elif(architecture == 'GRU'):
  predict = GRU(x_sequence, n_hidden)

# Define MSE cost and optimizer
cost = tf.reduce_mean(tf.squared_difference(predict, y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#### Define fuctions with use of tf session ####
def calculate_loss(sess, features, labels):
  iterations = int(features.shape[0] / batch_size)
  loss = 0.0
  p = 0
  for i in range(iterations):
    loss += sess.run(cost, feed_dict={x: features[p:p+batch_size], y: labels[p:p+batch_size]}).tolist()
    p += batch_size
  loss = loss / iterations
  return loss

def record_error(sess):
  train_loss = calculate_loss(sess, X_train, Y_train)
  test_loss = calculate_loss(sess, X_test, Y_test)
  log['train_loss'].append(train_loss)
  log['test_loss'].append(test_loss)
  print('train_loss = % .10f, test_loss = %.10f'  %(train_loss, test_loss))
  return np.mean([train_loss, test_loss])
  
# Define error compared with real price
def calculate_error_with_real_price(sess):
  real_prices = np.load('../data/data/ans_data.npz')
  real_last_two = real_prices['last_two'] #(2268, 1)
  real_last_one = real_prices['last_one'] #(2268, 1)
  X = np.concatenate((X_train, X_test), axis=0)
  Y = np.concatenate((Y_train, Y_test), axis=0)
  print('X shape = ', X.shape)

  predicted_log_return = sess.run(predict, feed_dict={x: X, y: Y}).tolist()
  predicted_last_one = np.exp( predicted_log_return + np.log(real_last_two) )
  print('predicted_last_one shape = ',  predicted_last_one.shape)
  print('predict price is: ', predicted_last_one[-5:])
  print('real price is: ',  real_last_one[-5:])
  
  MSE = np.mean((real_last_one - predicted_last_one)**2)
  print('MSE = %.10f' % MSE)


  '''
  iterations = int(features.shape[0] / batch_size)
  R_square = 0.0
  MSE = 0.0
  p = 0
  for i in range(iterations):
    predict = sess.run(predict, feed_dict={x: features[p:p+batch_size], y: labels[p:p+batch_size]}).tolist()
    p += batch_size
  '''

num_cur_best = 0
saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
  #init = tf.global_variables_initializer()
  #sess.run(init)

  # Restore model
  saver.restore(sess, 'models/tf_LSTM_3_best.ckpt')
  print("Model restored.")

  calculate_error_with_real_price(sess)


    
