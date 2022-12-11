import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.regularizers import l2


def myDNN(in_shape):
    """
    A function to create a Deep Neural Network model with custom 
    architecture for speech-music classification
    """
    model = Sequential()

    model.add(Dense(32, activation='relu', input_shape=in_shape))

    model.add(Dense(64, activation='relu', input_shape=in_shape))

    model.add(Dense(128, kernel_initializer=RandomUniform(minval=-0.05,
                    maxval=0.05), kernel_regularizer=l2(0.001),
                    activation='relu'))

    model.add(Dense(256, kernel_initializer=RandomUniform(minval=-0.05,
                    maxval=0.05), kernel_regularizer=l2(0.001), 
                    activation='relu'))

    model.add(Dense(512, kernel_initializer=RandomUniform(minval=-0.05,
                    maxval=0.05), kernel_regularizer=l2(0.001), 
                    activation='relu'))

    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))

    model.add(Dense(2, kernel_initializer=RandomUniform(minval=-0.05,
                    maxval=0.05), kernel_regularizer=l2(0.001),
                    activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
