from keras.models import Sequential
from keras.layers import Activation, Dense, Maximum
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import keras
import numpy as np

get_custom_objects().update({'sine': Activation(lambda x: K.sigmoid(10*K.sin(x)))})

n_rows = 5
limit = 5

model = Sequential()
model.add(Dense(units=(limit+1)*n_rows, activation='sine', input_dim=n_rows))
model.add(Dense(units=limit+1, activation='sine'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
npoints_x = 1000000

x_data_r = np.random.randint(0, 2**limit+1, (npoints_x//2,n_rows))
y_data_r = np.where(np.bitwise_xor.reduce(x_data_r,axis=1)==0, 0, 1).reshape((-1, 1))

x_data_0 = np.random.randint(0, 2**limit+1, (npoints_x//2,n_rows-1))
x_data_0 = np.hstack((x_data_0, np.bitwise_xor.reduce(x_data_0, axis=1).reshape((-1,1))))
y_data_0 = np.zeros((npoints_x//2, 1))

x_data = np.vstack((x_data_r, x_data_0))
y_data = np.vstack((y_data_r, y_data_0))

perm = np.random.permutation(npoints_x)
x_data = x_data[perm]
y_data = y_data[perm]

model.fit(x_data, y_data, epochs=10)

x_test_r = np.random.randint(0, 2**limit+1, (npoints_x//2,n_rows))
y_test_r = np.where(np.bitwise_xor.reduce(x_test_r,axis=1)==0, 0, 1).reshape((-1, 1))

x_test_0 = np.random.randint(0, 2**limit+1, (npoints_x//2,n_rows-1))
x_test_0 = np.hstack((x_test_0, np.bitwise_xor.reduce(x_test_0, axis=1).reshape((-1,1))))
y_test_0 = np.zeros((npoints_x//2, 1))

x_test = np.vstack((x_test_r, x_test_0))
y_test = np.vstack((y_test_r, y_test_0))

loss_and_metrics = model.evaluate(x_test, y_test)
y_pred = model.predict(x_data)