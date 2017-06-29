from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import json
from util import to_categorical, Dataset
import os

n_epochs = 10
batch_size = 32
show_steps = 3 # show statistics per k iters
learning_rate = 1e-7
n_hidden = 20 # 32
n_rnn_layers = 2
n_rnn_hidden = 100 #4
shuffle = True
USE_DROP = False

architecture = 'LSTM'
info = 'smallData-RMS1e7-Dense-2LSTM20-noDrop'
granularity = 'H6'  # 'M10'
data_name = 'rnn_MA_return_ans_3'#'rnn_normalized_MA_return.npz'  #'rnn_MA_return_candle' #'rnn_MA_return_candle' #'rnn_MA_return' #rnn_MA_return_ans_3
eid = granularity + '_' + info + '_' + data_name + '_ep' + str(n_epochs)
save_directory = 'models/' + eid
data_path = '../data/'+ granularity + '/' + data_name + '.npz'
ans_path ='../data/'+ granularity + '/' + 'rnn_trend_ans.npz'

log = {
    'experiment_id': eid,
    'train_loss': [],
    'test_loss': [],
    'train_accuracy': [],
    'test_accuracy': [],
    'n_rnn_hidden': n_rnn_hidden,
    'batch_size': batch_size, 
    'best_loss': 1, 
    'n_epochs': n_epochs,
    'shuffle': shuffle,
}
logfile = 'statistics/'+eid+'.json'

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
X_test = data['X'][1440:1440+64]
Y_test = data['Y'][1440:1440+64]

print('X_train[0]', X_train[0])
print('Y_train[0:10]', Y_train[0:10])

#Y_train = np.expand_dims(Y_train, axis=-1)
#Y_test = np.expand_dims(Y_test, axis=-1)
print('Y_train', Y_train.shape)
print('X_train', X_train.shape)

n_samples = X_train.shape[0]
n_timesteps = X_train.shape[1]
n_input = X_train.shape[2]
n_output = Y_train.shape[1]  # rise or fall
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
    print('rnn_outputs[-n_output:], ', len(rnn_outputs[-n_output:]))
    #predict = dense(rnn_outputs[-1], 1) # 100=>1
    predict = tf.stack([dense(rnn_outputs[-i], 1) for i in range(n_output, 0, -1)], axis=1)#dense2d(rnn_outputs[-n_output:], n_out=n_output) # (100, 3) => (3)
    print('predict, ', predict.get_shape())
    predict = predict[:,:,0]
    print('predict, ', predict.get_shape())
    # Define cost and optimizer
    cost = tf.reduce_mean(tf.squared_difference(predict, y))
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict) )
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)  #tf.train.AdamOptimizer(learning_rate).minimize(cost)

    last_timesteps = tf.stack(tf.unstack(x, axis=1)[-n_output:], axis=1)  # => (batch, feature_of_one_timestep)
    print('last_timesteps, ', last_timesteps.get_shape())
    last_MA = last_timesteps[:, :, 0]#tf.stack(tf.unstack(last_timesteps, axis=1)[0])  # MA is the first feature
    print('last_MA, ', last_MA.get_shape())
    predict_sign = tf.sign(predict-last_MA)
    real_sign = tf.sign(y-last_MA)
    correct_prediction = tf.equal(tf.sign(predict-last_MA), tf.sign(y-last_MA))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y, predict, cost, train_step, accuracy, predict_sign, real_sign


def calculate_loss(sess, features, labels):
    iterations = int(features.shape[0] / batch_size)
    avg_loss, avg_accuracy = 0.0, 0.0
    p = 0
    non_zero = 0#debug
    for i in range(iterations):
        cur_loss, cur_p, cur_psign, cur_rsign, cur_accuracy = sess.run([cost, predict, predict_sign, real_sign, accuracy], feed_dict={x: features[p:p+batch_size], y: labels[p:p+batch_size]})
        cur_p, cur_loss, cur_accuracy =cur_p.tolist(), cur_loss.tolist(), cur_accuracy.tolist()
        cur_psign, cur_rsign, = cur_psign.tolist(), cur_rsign.tolist()
        non_zero += np.count_nonzero(cur_p)
        #print('p sign:', cur_psign)
        #print('r sign:', cur_rsign)
        avg_loss += cur_loss
        avg_accuracy += cur_accuracy
    print('non-zero', non_zero / iterations)
    avg_loss = avg_loss / iterations
    avg_accuracy = avg_accuracy / iterations
    return avg_loss, avg_accuracy

def record_error(sess):
    train_loss, train_accuracy = calculate_loss(sess, X_train, Y_train)
    test_loss, test_accuracy = calculate_loss(sess, X_test, Y_test)
    log['train_loss'].append(train_loss)
    log['train_accuracy'].append(train_accuracy)
    log['test_loss'].append(test_loss)
    log['test_accuracy'].append(test_accuracy)
    print('train_loss = % .15f, test_loss = %.15f'  %(train_loss, test_loss), end='\t')
    print('train_accuracy = % .6f, test_accuracy = %.6f'  %(train_accuracy*100, test_accuracy*100))
    return np.mean([train_loss, test_loss])

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
    num_cur_best = 0
    x, y, predict, cost, train_step, accuracy, predict_sign, real_sign = model(drop=USE_DROP)
    saver = tf.train.Saver()
    # Launch the graph
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # Restore model
        #saver.restore(sess, 'models/H6_smallData-RMS1e5-Dense-2GRU100_rnn_MA_return_candle_ep100try/model.ckpt')
        #print("Model restored.")
        
        #print('Before train:\t', end="")
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
'''
    with open(statistcs_file, 'w') as f:
        json.dump(statistcs, f, indent=1)
'''
