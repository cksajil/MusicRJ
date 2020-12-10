# Import libraries
import os
import wave
import keras
import pyaudio
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.initializers import RandomUniform


# Set the streaming audio settings
CHUNK = 256
WIDTH = 1
CHANNELS = 1
RATE = 22050
RECORD_SECONDS = 30


# Define the Neural Network model Used
def create_model():

    model = Sequential()
    model.add(Dense(64, activation = 'relu', input_shape= (1, 256)))
    model.add(Dense(512, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05),
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        activation=tf.nn.relu))
    model.add(Dense(128, activation = 'relu', 
        kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dense (128, activation = 'relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(Dense (64, activation = 'relu', 
        kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dense(2, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05),
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        activation=tf.nn.softmax))

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Load trained model
modeltoDeploy = create_model()


# Loads the weights
checkpoint_path = "./TrainedModel/"
checkpoint_dir = os.path.dirname(checkpoint_path)
modeltoDeploy.load_weights(checkpoint_path)


# Create a probability models and labels for prediction
classes = ['Speech', 'Music']
probability_model = Sequential([modeltoDeploy, tf.keras.layers.Softmax()])


# Load a random file to test
testfiles = pd.read_csv('./Data/BasicData.csv')['Filename'].values
testfile = np.random.choice(testfiles)
wf = wave.open('./Data/Files/'+testfile, 'rb')


# Create a PyAudio handle to read, test, and play
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)


# Read data
data = wf.readframes(CHUNK)


# Test and play stream 
freqData = np.empty((0, CHUNK), int)
k = 0
while len(data) > 0:
    k+=1
    stream.write(data)
    idata = np.fromstring(data , np.int16)
    idata = np.abs(np.fft.fft(idata))
    T = np.sum(idata)/np.max(idata)
    if T>10:
        row = [list(idata)]
        if k%10 == 0:
            try:
                # pass
                predictions = probability_model.predict(row)
                # print(predictions)
                index = np.argmax(predictions[0])
                print(classes[index])
            except Exception as e:
                raise
    data = wf.readframes(CHUNK)


# Stop stream once done
stream.stop_stream()
stream.close()

# Close PyAudio handle
p.terminate()