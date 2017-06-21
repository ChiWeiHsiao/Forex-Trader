from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import json
from util import to_categorical, Dataset
import os

is_train = True
n_epochs = 100
batch_size = 32
show_steps = 10 # show statistics per 10 iters
learning_rate = 0.001
n_rnn_layers = 2
n_rnn_hidden = 10 #3
shuffle = True

architecture = 'LSTM'
granularity = 'H6'  # 'M10'
data_name = 'rnn_features'
eid = granularity + '_' + architecture + '_' + data_name + '_ep' + str(n_epochs)
save_directory = 'models/' + eid
data_path = '../data/'+ granularity + '/' + data_name + '.npz'

log = {
    'experiment_id': eid,
    'train_loss': [],
    'test_loss': [],
    'n_rnn_hidden': n_rnn_hidden,
    'batch_size': batch_size, 
    'best_loss': 1, 
    'n_epochs': n_epochs,
    'shuffle': shuffle,
}
logfile = 'statistics/'+eid+'.json'
print('id: ', eid)
print('num of epochs:', n_epochs)


#### Load data ####
#data = np.load('../data/data/rnn_data.npz')
data = np.load(data_path)
# 5 features: (log_return, upper_length, lower_length, whole_length, close_sub_open)
split_train_test = 5984  # divisible by batch_size
last_divisible_index = batch_size*int(data['X'].shape[0]/batch_size)
X_train = data['X'][:split_train_test]
Y_train = data['Y'][:split_train_test]
X_test  = data['X'][split_train_test:last_divisible_index]
Y_test  = data['Y'][split_train_test:last_divisible_index]
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
    return outputs[-1]
    
def LSTM(x_sequence, n_hidden):
    cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(cell, x_sequence, dtype=tf.float32)
    return outputs[-1]

def GRU(x_sequence, n_hidden):
    cell = rnn.GRUCell(n_hidden)
    outputs, states = rnn.static_rnn(cell, x_sequence, dtype=tf.float32)
    return outputs[-1]

def model(is_train):
    x = tf.placeholder('float', [None, n_timesteps, n_input])
    y = tf.placeholder('float', [None, n_output])
    if is_train:
        x = tf.nn.dropout(x, keep_prob=0.5)
    # Unstack to get a list of 'n_timesteps' tensors of shape (batch_size, n_input)
    x_sequence = tf.unstack(x, n_timesteps, 1)
    def lstm_cell():
        return rnn.BasicLSTMCell(n_rnn_hidden, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
    cell = lstm_cell
    if is_train:
        def cell():
            return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=0.5)
    stacked_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(n_rnn_layers)])

    init_state = stacked_cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = rnn.static_rnn(stacked_cell, x_sequence, initial_state=init_state, dtype=tf.float32)
    rnn_output = rnn_outputs[-1]

    W_fc = tf.Variable(tf.random_normal([n_rnn_hidden, n_output]))
    b_fc = tf.Variable(tf.random_normal([n_output]))
    h_fc = tf.matmul(rnn_output, W_fc) + b_fc
    predict = h_fc
    # Define MSE cost and optimizer
    cost = tf.reduce_mean(tf.squared_difference(h_fc, y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    return x, y, cost, train_step

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

if __name__ == '__main__':
    x, y, cost, train_step = model(is_train=is_train)
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
                # Save the model
                if log['best_loss'] - loss_this_iter > 0.0000000001:
                    num_cur_best += 1
                    print('Find and save %d current best loss model. %.10f' %(num_cur_best, loss_this_iter))
                    log['best_loss'] = loss_this_iter
                    if not os.path.exists(save_directory):
                        os.makedirs(save_directory)
                    save_path = saver.save(sess, '%s/model.ckpt' % (save_directory))

            # Shuffle data once for each epoch
            if shuffle and it % batch_size == 0:
                train_dataset.shuffle()
            
        # Save the model
        if log['best_loss'] > loss_this_iter:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            save_path = saver.save(sess, '%s/model.ckpt' % (save_directory))
            print('Best Model saved in file: %s' % save_path)


    # Print log to json file
    with open(logfile, 'w') as f:
        json.dump(log, f, indent=1)