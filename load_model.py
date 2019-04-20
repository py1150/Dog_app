from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# model architecture
VGG19_model = Sequential()

VGG19_model.add(Conv2D(32, kernel_size=(2, 2), activation='relu',input_shape=(7,7,512),padding='same'))
VGG19_model.add(MaxPooling2D((2, 2)))

VGG19_model.add(GlobalAveragePooling2D(input_shape=(7,7,512)))
VGG19_model.add(Dense(133, activation='softmax'))

VGG19_model.summary()

# compile
VGG19_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# load weights
VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')

