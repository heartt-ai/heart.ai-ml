import numpy as np
from scipy.io import wavfile
from scipy.signal import decimate
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, GlobalAvgPool1D, Dropout, BatchNormalization, Dense
from keras.regularizers import l2

CLASSES = ['null', 'normal', 'abnormal']

_, sample = wavfile.read("null.wav")
x_test = np.array(sample, dtype=np.float64)
x_test = np.resize(x_test, 396900)
x_test = np.array([x_test], dtype=np.float64)

x_test = decimate(x_test, 8, axis=1, zero_phase=True)
x_test = decimate(x_test, 8, axis=1, zero_phase=True)
x_test = decimate(x_test, 4, axis=1, zero_phase=True)

x_test = x_test / np.std(x_test, axis=1).reshape(-1,1)
x_test = x_test[:,:,np.newaxis]

model = Sequential()
model.add(Conv1D(filters=4, kernel_size=9, activation='relu',
                input_shape = (1551, 1),
                kernel_regularizer = l2(0.025)))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=4, kernel_size=9, activation='relu',
                kernel_regularizer = l2(0.05)))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=8, kernel_size=9, activation='relu',
                 kernel_regularizer = l2(0.1)))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=16, kernel_size=9, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.75))
model.add(GlobalAvgPool1D())
model.add(Dense(3, activation='softmax'))

model.load_weights('weights_gcloud.h5')

y_hat = model.predict(x_test)
print(CLASSES[np.argmax(y_hat)])