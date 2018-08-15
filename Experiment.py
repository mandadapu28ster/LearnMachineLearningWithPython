from random import seed
from random import randint
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy


# generate examples of random integers and their sum
def sum_pairs(n_numbers, largest):
    X, y = list(), list()
    dataset = numpy.loadtxt("AdditionData.csv", delimiter=",")
# split into input (X) and output (Y) variables
    X = dataset[:, 0:2]
    Y = dataset[:, 2]
    print(X, Y)
# format as NumPy arrays
    X,y = array(X), array(y)
# normalize
    X = X.astype('float') / float(largest * n_numbers)
    y = y.astype('float') / float(largest * n_numbers)
    return X, y


def sum_pairs_verify(n_numbers, largest):
    X, y = list(), list()
    dataset = numpy.loadtxt("AdditionData2Verify.csv", delimiter=",")
# split into input (X) and output (Y) variables
    X = dataset[:, 0:2]
    Y = dataset[:, 2]
    print(X, Y)
# format as NumPy arrays
    X,y = array(X), array(y)
# normalize
    X = X.astype('float') / float(largest * n_numbers)
    y = y.astype('float') / float(largest * n_numbers)
    return X, y


def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
	for i in range(n_examples):
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
		print(in_pattern, out_pattern)
		X.append(in_pattern)
		y.append(out_pattern)
	# format as NumPy arrays
	X,y = array(X), array(y)
	# normalize
	X = X.astype('float') / float(largest * n_numbers)
	y = y.astype('float') / float(largest * n_numbers)
	return X, y


# invert normalization
def invert(value, n_numbers, largest):
    return round(value * float(largest * n_numbers))


#fixed Values
n_examples = 16
n_numbers = 2
n_examples2verify = 16
largest = 100
# define LSTM configuration
n_batch = 1
n_epoch = 1
# create LSTM
model = Sequential()
model.add(LSTM(10, input_shape=(n_numbers, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# train LSTM
for _ in range(n_epoch):
    X, y = random_sum_pairs(n_examples,n_numbers, largest)
    X = X.reshape(n_examples, n_numbers, 1)
    model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2)


# evaluate on some new patterns
X, y = sum_pairs_verify(n_numbers, largest)
X = X.reshape(n_examples2verify, n_numbers, 1)
result = model.predict(X, batch_size=n_batch, verbose=0)


# calculate error
expected = [invert(x, n_numbers, largest) for x in y]
predicted = [invert(x, n_numbers, largest) for x in result[:, 0]]
rmse = sqrt(mean_squared_error(expected, predicted))
print('RMSE: %f' % rmse)
# show some examples
for i in range(20):
    error = expected[i] - predicted[i]
    print('Expected=%d, Predicted=%d (err=%d)' % (expected[i], predicted[i], error))