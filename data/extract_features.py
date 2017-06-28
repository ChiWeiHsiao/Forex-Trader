'''
(day, 10_min, candle) = (46(weeks)*4(days), 144(10min), 4(c,h,l,o))
'''
from datetime import datetime, timedelta
import numpy as np
import time 

def create_overlap_candles(infile_list, outfile, timesteps):
    is_first = True
    for infile in infile_list:
        year = np.load(infile)
        n_weeks = year.shape[0]
        n_candles_per_weeks = year.shape[1]
        for w in range(n_weeks):
            start_index = 0
            while start_index+timesteps <= n_candles_per_weeks:
                if is_first:
                    data = np.array([year[w][start_index:start_index+timesteps]])
                    print(data.shape)
                    is_first = False
                else:
                    data = np.append(data, np.array([year[w][start_index:start_index+timesteps]]), axis=0)
                start_index += 1
    np.save(outfile, data)
    print('Concatenated file saved as %s: ' %outfile, data.shape)
    n_samples = data.shape[0]
    print('Number of samples: %d' %n_samples)
    return n_samples

def extract_features(infile, n_samples, timesteps):
    candles = np.load(infile)
    # candles.shape = (n_samples, timesteps, 4) 
    # last dim is one candle: (c, h, l, o)
    # Log returns
    for i in range(timesteps+1):
        if i == 0:
            log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
            log_return = np.reshape(log_return, (-1,1))
            print(log_return.shape)
        else:
            new_log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
            new_log_return = np.reshape(new_log_return, (-1,1))
            log_return = np.append(log_return, new_log_return, axis=1)
    print('shape of log_return: ', log_return.shape) #(n_samples, timesteps+1)
    log_return = np.reshape(log_return, (n_samples, timesteps+1, 1))
    # upper_length = high_1 - max(open_3,close_0) => + 
    open_and_close = np.append(np.reshape(candles[:,:,3],(n_samples,timesteps+2,1)), np.reshape(candles[:,:,0], (n_samples,timesteps+2,1)), axis = 2)
    upper_length = candles[:,:,1] - np.amax(open_and_close, axis=2)
    upper_length = upper_length[:, 1:-1] # timesteps+2 -> timesteps
    upper_length = np.reshape(upper_length, (n_samples, timesteps, 1))
    # lower_length = min(open_3, close_0) - low => +
    open_and_close = np.append(np.reshape(candles[:,:,3],(n_samples,timesteps+2,1)), np.reshape(candles[:,:,0], (n_samples,timesteps+2,1)), axis = 2)
    lower_length = np.amin(open_and_close, axis=2) - candles[:,:,2]
    lower_length = lower_length[:, 1:-1]
    lower_length = np.reshape(lower_length, (n_samples, timesteps, 1))
    # whole_length = high_1 - low_2 => +
    whole_length = candles[:,:,1] - candles[:,:,2]
    whole_length = whole_length[:, 1:-1]
    whole_length = np.reshape(whole_length, (n_samples, timesteps, 1))
    # close_sub_open = close_0 - open_3 => +or-
    close_sub_open = candles[:,:,0] - candles[:,:,3]
    close_sub_open = close_sub_open[:, 1:-1]
    close_sub_open = np.reshape(close_sub_open, (n_samples, timesteps, 1))
    print('shape of close_sub_open: ', close_sub_open.shape)

    y = log_return[:,-1] # last one of log_return
    log_return = log_return[:,0:-1]
    x =  np.concatenate((log_return, upper_length, lower_length, whole_length, close_sub_open), axis=2)
    print('shape of x:', x.shape) #(n_samples, timesteps, 5)
    print('shape of y:', y.shape) #(n_samples, 1, 1)
    return x, y


def extract_candles_and_log_return(infile, n_samples, timesteps):
    candles = np.load(infile)
    # candles.shape = (n_samples, timesteps+2, 4) 
    # last dim is one candle: (c, h, l, o)
    for i in range(timesteps+1):
        if i == 0:
            log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
            log_return = np.reshape(log_return, (-1,1))
            print(log_return.shape)
        else:
            new_log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
            new_log_return = np.reshape(new_log_return, (-1,1))
            log_return = np.append(log_return, new_log_return, axis=1)
    print('shape of log_return: ', log_return.shape) #(n_samples, timesteps+1)
    log_return = np.reshape(log_return, (n_samples, timesteps+1, 1))
    # raw_candle
    raw_candle = candles[:, 1:-1]
    raw_candle = np.reshape(raw_candle, (n_samples, timesteps, 4))

    #y = log_return[:,-1] # last one of log_return
    last_one_close = candles[:,-1, 0]
    last_two_close = candles[:,-2, 0]
    #y = last_one_close - last_two_close
    rise_or_fall = np.where(last_one_close>last_two_close, 0, 1).astype(np.int32)
    y = to_categorical(rise_or_fall, 2)

    log_return = log_return[:,0:-1]  # remove last one 
    x =  np.concatenate((log_return, raw_candle), axis=2)
    print('shape of x:', x.shape)
    print('shape of y:', y.shape)
    return x, y

def extract_MA_features_answer(infile, timesteps, MA_window, normalize, candle):
    candles = np.load(infile)
    n_samples = candles.shape[0]
    # candles.shape = (n_samples, timesteps+2, 4) 
    # last dim is one candle: (c, h, l, o)
    for i in range(timesteps+1):
        if i == 0:
            log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
            log_return = np.reshape(log_return, (-1,1))
            print(log_return.shape)
        else:
            new_log_return = np.log(candles[:,i+1,0]) - np.log(candles[:,i,0])
            new_log_return = np.reshape(new_log_return, (-1,1))
            log_return = np.append(log_return, new_log_return, axis=1)
    print('shape of log_return: ', log_return.shape) #(n_samples, timesteps+1)
    log_return = np.reshape(log_return, (n_samples, timesteps+1, 1))

    # MA
    n_MA = (timesteps+1) - MA_window + 1
    is_first_sample = True
    for sample in range(log_return.shape[0]):
        for i in range(n_MA):
            cur_MA = np.mean(log_return[sample][i:i+MA_window])
            cur_return = log_return[sample][i+MA_window-1]
            cur = np.append(cur_MA, cur_return)
            if candle:
                cur_candle = candles[sample][i+MA_window]
                cur = np.append(cur, cur_candle)
            if i == 0:
                one_timestep_feature =  np.array([cur])
            elif i == n_MA-1:
                one_timestep_MA_ans = cur_MA
            else:
                one_timestep_feature = np.append(one_timestep_feature, np.array([cur]), axis=0)
        if is_first_sample:
            features = np.array([one_timestep_feature])
            answers = one_timestep_MA_ans
            is_first_sample = False
        else:
            features = np.append(features, np.array([one_timestep_feature]), axis=0)
            answers = np.append(answers, one_timestep_MA_ans)
    if normalize:
        ori_features = features
        features = (features/np.stack([features[:,0,:]+1e-10 for _ in range(features.shape[1])], axis=1)) - 1

        answers = (answers[:] / (ori_features[:,0,0]+1e-10)) - 1
    print('features: ', features.shape)
    print('answers: ', answers.shape)
    return features, answers


def create_dataset(x, y, split_index, outfile):
    # Split to Training set and Testing set
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    print('shape of x_train, y_train:', x_train.shape, y_train.shape)
    print('shape of x_test, y_test:', x_test.shape, y_test.shape)
    np.savez(outfile, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def create_answer(infile, outfile, n_samples):
    candles = np.load(infile)
    last_two = np.reshape(candles[:, -2, 0], (n_samples, 1))
    last_one = np.reshape(candles[:, -1, 0], (n_samples, 1))
    np.savez(outfile, last_two=last_two, last_one=last_one)

def to_categorical(y, nb_classes):
    y = np.reshape(y, y.shape[0])
    Y = np.zeros((len(y), nb_classes))
    Y[np.arange(len(y)), y] = 1.
    return Y

def create_trend_answer(infile, outfile, n_samples):
    candles = np.load(infile)
    last_two = np.reshape(candles[:, -2, 0], (n_samples, 1))
    last_one = np.reshape(candles[:, -1, 0], (n_samples, 1))
    rise_or_fall = np.where(last_one>last_two, 0, 1).astype(np.int32)
    to_one_hot = to_categorical(rise_or_fall, 2)
    np.savez(outfile, to_one_hot)    



if __name__ == '__main__':
    timesteps = 12
    '''
    year_raw_candle_files = []
    for i in range(2005, 2017+1):
        year_raw_candle_files.append('H6/candles/raw_candles_%s.npy' %i)
    n_samples = create_overlap_candles(year_raw_candle_files, 'H6/H6-overlap-'+str(timesteps+2), timesteps+2)
    '''
    #n_samples = 6237
    infile = 'H6/H6-overlap-14.npy'  #'candles_05-17.npy'
    x, y = extract_MA_features_answer(infile, timesteps, MA_window=5, normalize=True, candle=False)
    np.savez('H6/rnn_normalized_MA_return', X=x, Y=y)
    
    #x, y = extract_features(infile, n_samples, timesteps)
    #np.savez('H6/rnn_features', X=x, Y=y)
    #create_dataset(x, y, split_index=6000, outfile='H6/rnn_features')

    #x, y = extract_candles_and_log_return(infile, n_samples, timesteps)
    #np.savez('H6/rnn_candles_return', X=x, Y=y) 
    #create_dataset(x, y, split_index=6000, outfile='H6/rnn_raw_features')
    
    #create_answer(infile, outfile='H6/rnn_ans', n_samples)
    #create_trend_answer(infile, 'H6/rnn_trend_ans', n_samples)