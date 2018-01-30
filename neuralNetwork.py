from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset from local file
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]


# load pima indians dataset from url
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
dataset = pandas.read_csv(url, delimiter=",")
# Split-out validation dataset
array = dataset.values
X = array[:,0:8]
Y = array[:,8]

# shape; check data
print(dataset.shape)

#Processing starting
print("Processing started ======= >")

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model; these lines are ment for initial run only
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# calculate predictions; these lines makes complete E2E
print("Real stuff comes from here")
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


