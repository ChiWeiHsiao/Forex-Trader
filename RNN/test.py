from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import json
from util import to_categorical, Dataset
import os

n_epochs = 3
batch_size = 32
show_steps = 3 # show statistics per k iters
learning_rate = 1e-7
n_hidden = 20 # 32
n_rnn_layers = 2
n_rnn_hidden = 100 #4
shuffle = False
USE_DROP = False

architecture = 'LSTM'
info = 'smallData-RMS1e7-Dense-2LSTM20-noDrop'  #'smallData-RMS1e7-Dense-2LSTM20-noDrop'
granularity = 'H6'  # 'M10'
data_name = 'rnn_MA_return' #'rnn_MA_return_candle' #'rnn_MA_return'
eid = granularity + '_' + info + '_' + data_name + '_ep' + str(n_epochs)
save_directory = 'models/' + eid
data_path = '../data/'+ granularity + '/' + data_name + '.npz'
ans_path ='../data/'+ granularity + '/' + 'rnn_trend_ans.npz'


statistcs = {
    'real_price': [],
    'predict_price': [],
    'sign_accuracy': [],
    'MSE': 0.0,
    'R2': 0.0,
    'real': [],
    'predict:': [],
}
statistcs_file = 'statistics/' + eid +'_train.json'

print('id: ', eid)
print('num of epochs:', n_epochs)


#### Load data ####
#data = np.load('../data/data/rnn_data.npz')
data = np.load(data_path)
# 5 features: (log_return, upper_length, lower_length, whole_length, close_sub_open)
split_train_test = 120*32  # divisible by batch_size
last_divisible_index = batch_size*int(data['X'].shape[0]/batch_size)
X_train = data['X'][:split_train_test]
Y_train = data['Y'][:split_train_test]
X_test  = data['X'][split_train_test:last_divisible_index]
Y_test  = data['Y'][split_train_test:last_divisible_index]

# Try smaller data
X_train = data['X'][960:1440]
Y_train = data['Y'][960:1440]
X_test = data['X'][1440:1440+32]
Y_test = data['Y'][1440:1440+32]


Y_train = np.expand_dims(Y_train, axis=-1)
Y_test = np.expand_dims(Y_test, axis=-1)
print('Y_train', Y_train.shape)
print('X_train', X_train.shape)

n_samples = X_train.shape[0]
n_timesteps = X_train.shape[1]
n_input = X_train.shape[2]
n_output = 1  # rise or fall
n_iters = int(n_epochs * n_samples / batch_size)
print('number of iterations %d' %n_iters)
# Convert to Dataset instance 
train_dataset = Dataset(X_train, Y_train, batch_size)


def dense(x, n_out):
    w_init = tf.random_normal_initializer(mean=0., stddev=0.3, dtype=tf.float32)
    b_init = tf.constant_initializer(0.)
    return tf.contrib.layers.fully_connected(x, n_out, activation_fn=None, weights_initializer=w_init, biases_initializer=b_init)

def dense2d(x, n_out):
    x_sequence = tf.unstack(x, n_timesteps, 1)
    #print(tf.unstack(x_sequence[0], batch_size, 0)[0].get_shape())
    timesteps = [dense(x_sequence[t], n_out) for t in range(n_timesteps)]
    return timesteps

def model(drop):
    x = tf.placeholder('float', [None, n_timesteps, n_input])
    y = tf.placeholder('float', [None, n_output])
    b_size = tf.shape(x)[0]
    if drop:
        drop_x = tf.nn.dropout(x, keep_prob=0.5)
        x_sequence = dense2d(drop_x, n_out=n_hidden)
    else:
        x_sequence = dense2d(x, n_out=n_hidden)
    # Unstack to get a list of 'n_timesteps' tensors of shape (batch_size, n_input)
    #x_sequence = tf.unstack(x, n_timesteps, 1)
    def lstm_cell():
        return rnn.BasicLSTMCell(n_rnn_hidden, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
    def rnn_cell():
        return rnn.BasicRNNCell(n_rnn_hidden, reuse=tf.get_variable_scope().reuse)
    def gru_cell():
        return rnn.GRUCell(n_rnn_hidden, reuse=tf.get_variable_scope().reuse)
    cell_selection = {'LSTM': lstm_cell, 'RNN': rnn_cell, 'GRU': gru_cell}
    cell = cell_selection[architecture]#lstm_cell
    if drop:
        def cell():
            return tf.contrib.rnn.DropoutWrapper(cell_selection[architecture](), output_keep_prob=0.5)
            #return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=0.5)
    stacked_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(n_rnn_layers)])

    init_state = stacked_cell.zero_state(b_size, tf.float32)
    rnn_outputs, final_state = rnn.static_rnn(stacked_cell, x_sequence, initial_state=init_state, dtype=tf.float32)
    predict = dense(rnn_outputs[-1], n_output)
    # Define cost and optimizer
    cost = tf.reduce_mean(tf.squared_difference(predict, y))
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict) )
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)  #tf.train.AdamOptimizer(learning_rate).minimize(cost)

    last_timesteps = tf.unstack(x, axis=1)[-1]  # => (batch, feature_of_one_timestep)
    last_MA = tf.unstack(last_timesteps, axis=1)[0]  # MA is the first feature
    predict_sign = tf.sign(predict-last_MA)
    real_sign = tf.sign(y-last_MA)
    correct_prediction = tf.equal(tf.sign(predict-last_MA), tf.sign(y-last_MA))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y, predict, cost, train_step, accuracy, predict_sign, real_sign


def statistcs_test(sess, X, Y, statistcs):
    # Predict prices
    dataset = Dataset(X, Y, batch_size)
    real_MA = Y
    n_batch = int(X.shape[0]/batch_size)
    for i in range(n_batch):
        next_x, next_y = dataset.next_batch()
        if i == 0:
            #predict_log_return = sess.run(predict, feed_dict={x: next_x, y: next_y}).tolist()
            print('next y: ', next_y.shape)
            predict_MA, sign_accuracy = sess.run([predict, accuracy], feed_dict={x: next_x, y: next_y})
            predict_MA = predict_MA
            sign_accuracy = np.array([sign_accuracy])
        else:
            #predict_log_return = np.append(predict_log_return, sess.run(predict, feed_dict={x: next_x, y: next_y}).tolist(), axis=0)
            ma, acc = sess.run([predict, accuracy], feed_dict={x: next_x, y: next_y})
            predict_MA = np.append(predict_MA, ma, axis=0)
            sign_accuracy = np.append(sign_accuracy, np.array([acc]), axis=0)
    #predict_last_one = np.exp( predict_log_return + np.log(real_last_two) )
    # Export prices to statistcs_file
    statistcs['real_price'] = real_MA.tolist()
    statistcs['predict_price'] = predict_MA.tolist()
    statistcs['sign_accuracy'] = sign_accuracy.tolist()

    # Calculate MSE
    MSE = np.mean((real_MA - predict_MA)**2)
    statistcs['MSE'] = MSE
    print('MSE = %.10f' % MSE)

    # Caluculate Variance
    mean_real = np.mean(real_MA)
    var_real = np.var(real_MA)
    print('Real mean=%.15f, var=%.15f' %(mean_real, var_real))
    mean_predict = np.mean(predict_MA)
    var_predict = np.var(predict_MA)
    print('predict MA: ', predict_MA)
    print('Predict mean=%.25f, var=%.25f' %(mean_predict, var_predict))

    # Calculate R^2
    SSR = np.sum(np.square(predict_MA - mean_real))
    SST = np.sum(np.square(real_MA - mean_real))
    R2 = SSR / SST
    statistcs['R2'] = R2
    print('R^2=%.4f' %R2)


if __name__ == '__main__':
    x, y, predict, cost, train_step, accuracy, predict_sign, real_sign = model(drop=USE_DROP)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore best model to perform statistics test
        saver.restore(sess, '%s/model.ckpt' % save_directory)
        print("Model restored: %s" % save_directory)
        statistcs_test(sess, X_test, Y_test, statistcs)

    with open(statistcs_file, 'w') as f:
        json.dump(statistcs, f, indent=1)