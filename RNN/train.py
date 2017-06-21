from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import json
from util import to_categorical, Dataset
import os

is_train = True
n_epochs = 30
batch_size = 32
show_steps = 10 # show statistics per 10 iters
learning_rate = 0.001
n_hidden = 32
n_rnn_layers = 2
n_rnn_hidden = 10 #4
shuffle = True

architecture = 'Dense-2LSTM'
granularity = 'H6'  # 'M10'
data_name = 'rnn_candles_return' #'rnn_candles'
eid = granularity + '_' + architecture + '_' + data_name + '_ep' + str(n_epochs) + 'try'
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

#Y_train = np.expand_dims(Y_train, axis=-1)
#Y_test = np.expand_dims(Y_test, axis=-1)
print('Y_train:', Y_test[0:10])
n_samples = X_train.shape[0]
n_timesteps = X_train.shape[1]
n_input = X_train.shape[2]
n_output = 2  # rise or fall
n_iters = int(n_epochs * n_samples / batch_size)
print('number of iterations %d' %n_iters)
# Convert to Dataset instance 
train_dataset = Dataset(X_train, Y_train, batch_size)


def dense(x, n_out):
    return tf.contrib.layers.fully_connected(x, n_out, activation_fn=tf.nn.relu)

def dense2d(x, n_out):
    x_sequence = tf.unstack(x, n_timesteps, 1)
    #print(tf.unstack(x_sequence[0], batch_size, 0)[0].get_shape())
    timesteps = [dense(x_sequence[t], n_out) for t in range(n_timesteps)]
    return timesteps

def model(is_train):
    x = tf.placeholder('float', [None, n_timesteps, n_input])
    y = tf.placeholder('float', [None, n_output])
    if is_train:
        x = tf.nn.dropout(x, keep_prob=0.5)

    x_sequence = dense2d(x, n_out=n_hidden)
    # Unstack to get a list of 'n_timesteps' tensors of shape (batch_size, n_input)
    #x_sequence = tf.unstack(x, n_timesteps, 1)
    def lstm_cell():
        return rnn.BasicLSTMCell(n_rnn_hidden, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
    cell = lstm_cell
    if is_train:
        def cell():
            return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=0.5)
    stacked_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(n_rnn_layers)])

    init_state = stacked_cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = rnn.static_rnn(stacked_cell, x_sequence, initial_state=init_state, dtype=tf.float32)
    predict = dense(rnn_outputs[-1], n_output)
    # Define cost and optimizer
    #cost = tf.reduce_mean(tf.squared_difference(predict, y))
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict) )
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y, cost, train_step, accuracy


x, y, cost, train_step, accuracy = model(is_train=is_train)


def calculate_loss(sess, features, labels):
    iterations = int(features.shape[0] / batch_size)
    avg_loss, avg_accuracy = 0.0, 0.0
    p = 0
    for i in range(iterations):
        cur_loss, cur_accuracy = sess.run([cost, accuracy], feed_dict={x: features[p:p+batch_size], y: labels[p:p+batch_size]})
        cur_loss, cur_accuracy = cur_loss.tolist(), cur_accuracy.tolist()
        avg_loss += cur_loss
        avg_accuracy += cur_accuracy
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
    print('train_loss = % .10f, test_loss = %.10f'  %(train_loss, test_loss), end='\t')
    print('train_accuracy = % .4f, test_accuracy = %.4f'  %(train_accuracy, test_accuracy))
    return np.mean([train_loss, test_loss])

if __name__ == '__main__':
    num_cur_best = 0
    saver = tf.train.Saver()
    # Launch the graph
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # Restore model
        #saver.restore(sess, 'models/try_2.ckpt')
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

    print('Average Train Accuracy = ', sum(log['train_accuracy']) / len(log['train_accuracy']))
    print('Average Test Accuracy = ', sum(log['test_accuracy']) / len(log['test_accuracy']))

    # Print log to json file
    with open(logfile, 'w') as f:
        json.dump(log, f, indent=1)