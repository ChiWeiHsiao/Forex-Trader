''' data utility functions '''
import numpy as np

def shuffle(*pairs):
  pairs = list(pairs)
  for i, pair in enumerate(pairs):
    pairs[i] = np.array(pair)
  p = np.random.permutation(len(pairs[0]))
  return tuple(pair[p] for pair in pairs)

def to_categorical(y, nb_classes):
  y = np.asarray(y, dtype='int32')
  Y = np.zeros((len(y), nb_classes))
  Y[np.arange(len(y)),y] = 1.
  return Y

class Dataset():
  def __init__(self, X, Y, batch_size):
    self.X = X
    self.Y = Y
    self.X_batch, self.Y_batch = [], []
    self.batch_size = batch_size
    self.state = 0
    self.total_batch = int(X.shape[0] / batch_size)

  def next_batch(self):
    start = self.state * self.batch_size
    end = start + self.batch_size
    next_x = self.X[start:end]
    next_y = self.Y[start:end]
    self.state += 1
    self.state %= self.total_batch
    return next_x, next_y

  def shuffle(self):
    self.X, self.Y = shuffle(self.X, self.Y)


