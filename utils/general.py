import yaml
import librosa
import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm
from utils.dnn import myDNN
from utils.cnn import myCNN


def load_config(config_name):
    CONFIG_PATH = "./config/"
    with open(path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("my_config.yaml")


def strided_app(a, L, S):
    """
    Function to split audio as overlapping segments
    L : Window length
    S : Stride (L/stepsize)
    """
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S*n, n))


def create_model(in_shape):
    if config['default_model'] == 'myDNN':
        model = myDNN(in_shape)
    else:
        model = myCNN(in_shape)
    return model


def mapLabels(x):
    if x == 'Music':
        flag = 1
    else:
        flag = 0
    return flag


def generateDNNData(basicdata, frameS, offset):
    CorpusPath = path.join(config["data_directory"], config["file_directory"])
    Corpusfiles = basicdata['Filename'].values
    spectrumData = np.empty((0, frameS), int)
    spectrumCol = ['S'+str(i) for i in range(frameS)]
    i = 0
    Fcol = []

    for file in tqdm(Corpusfiles):
        i += 1
        x, Fs = librosa.load(CorpusPath+file)
        segments = strided_app(x, L=frameS, S=offset)
        for _ in range(64):
            ids = np.random.randint(10, size=2)
            for segment in segments[ids, :]:
                Fcol.append(file)
                row = np.abs(np.fft.fft(segment)) #256d vector
                spectrumData = np.vstack([spectrumData, row])
    spectrumDF = pd.DataFrame(spectrumData, columns=spectrumCol)
    spectrumDF['Filename'] = Fcol

    masterdata = pd.merge(basicdata, spectrumDF, on='Filename')
    masterdata.to_csv(path.join(config["data_directory"],
                                config["master_data"]), index=False)


def generateCNNData(basicdata, frameS, offset):
    CorpusPath = path.join(config["data_directory"], config["file_directory"])
    labels = []
    segmentData = []
    for index, row in tqdm(basicdata.iterrows(), total=basicdata.shape[0]):
        x, Fs = librosa.load(CorpusPath+row['Filename'])
        spectro_data = np.abs(librosa.stft(x))
        segments = np.array_split(spectro_data, 10)
        for segment in segments[:2]:
            if segment.shape[0] < 103:
                pad_row = np.zeros((1, 1292))
                segment_padded = np.vstack((segment, pad_row))
                segmentData.append(segment_padded)
            else:
                segmentData.append(segment)
            flag = mapLabels(row['Label'][0])
            labels.append(flag)
    cnn_features_data_path = path.join(config["data_directory"],
                                       config["cnn_X_data"])
    cnn_labels_data_path = path.join(config["data_directory"],
                                     config["cnn_y_data"])

    np.save(cnn_features_data_path, segmentData, allow_pickle=True)
    np.save(cnn_labels_data_path, labels, allow_pickle=True)
