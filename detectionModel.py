import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

# Splitting to training data and test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(X_train.shape)
# Reshaping the training and test data from 3d to 4d for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')

# print(X_train.shape)
# print(X_train)
# print(y_train)
# Converting each element value between 0 and 1
X_train/=255
X_test/=255
# Conversion of y train and y test to one hot encoding
number_of_classes = 10
y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)
# print(y_train)

# create model
model = Sequential()
# The first hidden layer with 32 filters/channels,size of 5x5 and activation function
""" Input layer """
model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
# To reduce overfitting which means that too much of training can turn out to be a disadvantage as it looks for more specific details
model.add(MaxPooling2D(pool_size=(2, 2)))
""" This is the next hidden layer  """
model.add(Conv2D(32, (3, 3), activation='relu'))
# One more maxpool
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout 20% neurons to prevent the overfitting problem
model.add(Dropout(0.2))
# To convert 2d matrix data to 1d matrix data
model.add(Flatten())
""" THe last layer  with 128 neurons """
model.add(Dense(128, activation='relu'))
""" This is the output layer which uses 10 neurons and activation function as softmax """
# Softmax is for multi class classifcation and sigmoid is for binary classification
model.add(Dense(number_of_classes, activation='softmax'))
""" Compilation of the model """
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
""" Now the training begins  """
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=200)
# Save the model
model.save('models/mnistCNN.h5')
""" Evaluation of the model """
metrics = model.evaluate(X_test, y_test, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)