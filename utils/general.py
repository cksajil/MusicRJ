import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from keras.optimizers import RMSprop

def strided_app(a, L, S):  
    """
    Function to split audio as overlapping segments
    L : Window length
    S : Stride (L/stepsize)
    """
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a,\
        shape=(nrows,L),\
        strides=(S*n,n))

def create_model():
	model = Sequential()
	model.add(Dense(64,\
		activation = 'relu',\
		input_shape = X_train[0].shape))
	model.add(Dense(512,\
		kernel_initializer = RandomUniform(minval=-0.05, maxval=0.05),\
		kernel_regularizer = tf.keras.regularizers.l2(0.001),\
		activation = tf.nn.relu))
	model.add(Dense(128,\
		activation = 'relu',\
		kernel_regularizer = tf.keras.regularizers.l2(0.001)))
	model.add(Dense(128,\
		activation = 'relu',\
		kernel_regularizer = tf.keras.regularizers.l2(0.001)))
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(Dense(64,\
		activation = 'relu',\
		kernel_regularizer = tf.keras.regularizers.l2(0.001)))
	model.add(Dense(2,\
		kernel_initializer = RandomUniform(minval=-0.05, maxval=0.05),\
		kernel_regularizer = tf.keras.regularizers.l2(0.001),\
	 	activation = tf.nn.softmax))
	model.compile(optimizer = RMSprop(learning_rate=0.0001),\
		loss='sparse_categorical_crossentropy',\
		metrics=['accuracy'])
	return model