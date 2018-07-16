from random import seed
from random import randint

from numpy import prod

seed(1)
X, y = list(), list()
for i in range(100):
	in_pattern = [randint(1,100) for _ in range(2)]
	out_pattern = prod(in_pattern)
	print(in_pattern, out_pattern)
	# X.append(in_pattern)
	# y.append(out_pattern)

# # load pima indians dataset from local file
# dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]
# print(X, Y)