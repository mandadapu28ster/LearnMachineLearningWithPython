from random import seed
from random import randint
import numpy
from numpy import prod

seed(3)
X, y = list(), list()
for i in range(10):
	in_pattern = [randint(1,100) for _ in range(2)]
	out_pattern = sum(in_pattern)
	print(in_pattern, out_pattern)
	# X.append(in_pattern)
	# y.append(out_pattern)

# load pima indians dataset from local file
# dataset = numpy.loadtxt("AdditionData.csv", delimiter=",")
# split into input (X) and output (Y) variables
# X = dataset[:,0:2]
# Y = dataset[:,2]
# print(X, Y)