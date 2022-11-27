# Import libraries
import wave
import pyaudio
import numpy as np
import pandas as pd
from utils.general import create_model
from keras.models import Sequential
from tensorflow.keras.layers import Softmax

# Set the streaming audio settings
CHUNK = 256
WIDTH = 1
CHANNELS = 1
RATE = 32000
RECORD_SECONDS = 30

# Initialize model
modeltoDeploy = create_model((CHUNK,))

# Loads the weights
file_path = "./TrainedModel/best_model.hdf5"
modeltoDeploy.load_weights(file_path)

# Create a probability models and labels for prediction
classes = ['Speech', 'Music']
probability_model = Sequential([modeltoDeploy, Softmax()])

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
    stream.write(data)
    idata = np.fromstring(data, np.int16)
    idata = np.abs(np.fft.fft(idata))
    threshold = np.sum(idata)
    row = [list(idata)]
    if k % 17 == 0:
        if threshold < 1000000:
            print('Silence')
        else:
            predictions = probability_model.predict(row)
            index = np.argmax(predictions[0])
            print(classes[index])
    k += 1
    data = wf.readframes(CHUNK)

# Stop stream once done
stream.stop_stream()
stream.close()

# Close PyAudio handle
p.terminate()
