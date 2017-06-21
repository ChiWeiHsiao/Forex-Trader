from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import json
from util import to_categorical, Dataset

data_dir = 'H6/'
data_name = data_dir + 'rnn_features.npz'
ans_name = data_dir + 'rnn_ans.npz'
model_dir = 'H6_LSTM_rnn_features_ep300'
print('Model: %s' %model_dir)
print('Features Data: %s' %data_name)

architecture = 'LSTM'
is_train = False
batch_size = 32
learning_rate = 0.001
n_rnn_layers = 2
n_rnn_hidden = 10
shuffle = True

statistcs = {
    'real_price': [],
    'predict_price': [],
    'sign_accuravy': 0.0,
    'MSE': 0.0,
}
statistcs_file = 'statistics/' + model_dir +'_train.json'

test_statistcs = {
    'real_price': [],
    'predict_price': [],
    'sign_accuravy': 0.0,
    'MSE': 0.0,
}
test_statistcs_file = 'statistics/' + model_dir +'_test.json'


#### Load data ####
data = np.load('../data/{}'.format(data_name))
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
    return x, y, predict, cost, train_step


# Define error compared with real price
def calculate_error_with_real_price(sess, predict_op, X, Y, real_last_one, real_last_two, statistcs):
    # Predict prices
    dataset = Dataset(X, Y, batch_size)
    n_batch = int(X.shape[0]/batch_size)
    for i in range(n_batch):
        next_x, next_y = dataset.next_batch()
        if i == 0:
            predict_log_return = sess.run(predict, feed_dict={x: next_x, y: next_y}).tolist()
        else:
            predict_log_return = np.append(predict_log_return, sess.run(predict, feed_dict={x: next_x, y: next_y}).tolist(), axis=0)

    predict_last_one = np.exp( predict_log_return + np.log(real_last_two) )
    # Export prices to statistcs_file
    statistcs['real_price'] = real_last_one.tolist()
    statistcs['predict_price'] = predict_last_one.tolist()

    # Show last 5 statistics
    print('last price is: ',  real_last_two[-5:])
    print('predict price is: ', predict_last_one[-5:])
    print('real price is: ',  real_last_one[-5:])

    # Calculate MSE
    MSE = np.mean((real_last_one - predict_last_one)**2)
    statistcs['MSE'] = MSE
    print('MSE = %.10f' % MSE)

    # Caluculate Variance
    mean_real = np.mean(real_last_one)
    var_real = np.var(real_last_one)
    print('Real mean=%.4f, var=%.5f' %(mean_real, var_real))
    mean_predict = np.mean(predict_last_one)
    var_predict = np.var(predict_last_one)
    print('Predict mean=%.4f, var=%.5f' %(mean_predict, var_predict))

    length = real_last_one.shape[0]
    real_last_one = np.reshape(real_last_one, length)
    real_last_two = np.reshape(real_last_two, length)
    predict_last_one = np.reshape(predict_last_one, length)

    # Calculate sign accuracy
    real_diff = real_last_one - real_last_two
    predict_diff = predict_last_one - real_last_two
    real_diff_sign = np.sign(real_diff)
    predict_diff_sign = np.sign(predict_diff)
    print('real_diff: ', real_diff[-20:])
    print('predict_diff: ', predict_diff[-20:])
    num_correct_sign = np.count_nonzero(np.equal(real_diff_sign, predict_diff_sign))
    sign_accuracy = num_correct_sign / real_diff_sign.shape[0]
    statistcs['sign_accuracy'] = sign_accuracy
    print('Sign Accuracy = %.10f' % sign_accuracy)
    


x, y, predict, cost, train_step = model(is_train)
num_cur_best = 0
saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    # Restore model
    saver.restore(sess, 'models/'+model_dir+'/model.ckpt')
    print("Model restored.")

    # Trainset
    real_prices = np.load('../data/{}'.format(ans_name))
    real_last_two = real_prices['last_two'][:split_train_test]
    real_last_one = real_prices['last_one'][:split_train_test]
    # X = np.concatenate((X_train, X_test), axis=0)
    print('========== Train ==========')
    X = X_train
    Y = Y_train
    print('X shape = ', X.shape)
    calculate_error_with_real_price(sess, predict, X, Y, real_last_one, real_last_two, statistcs)
    
    print('========== Test ==========')
    real_last_two = real_prices['last_two'][split_train_test:last_divisible_index]
    real_last_one = real_prices['last_one'][split_train_test:last_divisible_index]
    X = X_test
    Y = Y_test
    print('X shape = ', X.shape)
    calculate_error_with_real_price(sess, predict, X, Y, real_last_one, real_last_two, test_statistcs)
    
# Print statistcs to json file

with open(statistcs_file, 'w') as f:
    json.dump(statistcs, f, indent=1)
with open(test_statistcs_file, 'w') as f:
    json.dump(test_statistcs, f, indent=1)
        
