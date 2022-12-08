# Import libraries
import wave
import pyaudio
import numpy as np
import pandas as pd
from os import path
from keras.models import Sequential
from tensorflow.keras.layers import Softmax
from utils.general import create_model, load_config


def main():

    config = load_config("my_config.yaml")

    # Set the streaming audio settings
    CHUNK = config['FRAMES']

    # Initialize model
    modeltoDeploy = create_model((CHUNK,))
    print(modeltoDeploy.summary())

    # Select Model and Loads the weights
    if config['default_model'] == 'myDNN':
        file_path = path.join(config["model_directory"],
                              config["dnn_model_name"])
    else:
        file_path = path.join(config["model_directory"],
                              config["cnn_model_name"])
    
    modeltoDeploy.load_weights(file_path)

    # Create a probability models and labels for prediction
    classes = ['Speech', 'Music']
    probability_model = Sequential([modeltoDeploy, Softmax()])

    # Load a random file to test
    random_file_path = path.join(config["data_directory"],
                                 config["basic_data"])
    testfiles = pd.read_csv(random_file_path)['Filename'].values
    testfile = np.random.choice(testfiles)

    audio_file_dir = path.join(config["data_directory"],
                               config["file_directory"])
                               
    wf = wave.open(audio_file_dir+testfile, 'rb')

    # Create a PyAudio handle to read, test, and play
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read data
    data = wf.readframes(CHUNK)

    # Test and play stream
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


if __name__ == '__main__':
    main()
