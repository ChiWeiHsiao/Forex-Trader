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
  
num_cur_best = 0
saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  # Restore model
  #saver.restore(sess, 'models/try_2.ckpt')
  #print("Model restored.")
  print('Before train:\t', end="")
  record_error(sess)
  for it in range(n_iters):
    # Train next batch
    next_x, next_y = train_dataset.next_batch()
    sess.run(train_step, feed_dict={x: next_x, y: next_y})
    # Record loss
    if it % show_steps == 0:
      print('Iterations %4d:\t' %(it+1) , end="")
      loss_this_iter = record_error(sess)
      if log['best_loss'] - loss_this_iter > 0.0000000001:
        num_cur_best += 1
        print('Find %d current best loss! %.10f' %(num_cur_best, loss_this_iter))
        log['best_loss'] = loss_this_iter
        save_path = saver.save(sess, 'models/%s.ckpt' % (eid+'_best'))

    # Shuffle data once for each epoch
    if shuffle and it % batch_size == 0:
      train_dataset.shuffle()
    
  # Save the model
  save_path = saver.save(sess, 'models/%s.ckpt' % eid)
  print('Model saved in file: %s' % save_path)


# Print log to json file
with open(logfile, 'w') as f:
  json.dump(log, f, indent=1)
