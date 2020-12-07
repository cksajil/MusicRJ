import os
import pyaudio
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.initializers import RandomUniform

CHUNK = 256
WIDTH = 2
CHANNELS = 1
RATE = 48000
RECORD_SECONDS = 1


def create_model():

	model = Sequential()
	model.add(Dense(64, activation = 'relu', input_shape= (1, 256)))
	model.add(Dense(512, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05),
		kernel_regularizer=tf.keras.regularizers.l2(0.001),
	 	activation=tf.nn.relu))
	model.add(Dense(128, activation = 'sigmoid', 
		kernel_regularizer=tf.keras.regularizers.l2(0.001)))
	model.add(Dense (128, activation = 'relu',
		kernel_regularizer=tf.keras.regularizers.l2(0.001)))
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(Dense (64, activation = 'sigmoid', 
		kernel_regularizer=tf.keras.regularizers.l2(0.001)))
	model.add(Dense(2, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05),
		kernel_regularizer=tf.keras.regularizers.l2(0.001),
	 	activation=tf.nn.softmax))

	model.compile(optimizer='adam',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	return model


# Load Saved Model
modeltoDeploy = create_model()


# Loads the weights
checkpoint_path = "./TrainedModel/"
checkpoint_dir = os.path.dirname(checkpoint_path)
modeltoDeploy.load_weights(checkpoint_path)

classes = ['Female', 'Male']

probability_model = Sequential([modeltoDeploy, tf.keras.layers.Softmax()])



p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                input_device_index=7,
                frames_per_buffer=CHUNK)

print("* recording")

freqData = np.empty((0, CHUNK), int)
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    data = np.fromstring(data , np.int16)
    data = np.abs(np.fft.fft(data))
    # freqData = np.vstack([freqData,data])
    # data = np.mean(freqData, axis=0)
    row = [list(data)]
    try:
    	predictions = probability_model.predict(row)
    	index = np.argmax(predictions[0])
    	print(classes[index])
    except Exception as e:
    	raise
    else:
    	pass
    finally:
    	pass
    
    # stream.write(data, CHUNK)



stream.stop_stream()
stream.close()

p.terminate()