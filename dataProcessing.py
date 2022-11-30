import yaml
import librosa
import numpy as np
from os import path
import pandas as pd
from tqdm import tqdm
from utils.general import strided_app


def main():

    CONFIG_PATH = "./config/"

    def load_config(config_name):
        with open(path.join(CONFIG_PATH, config_name)) as file:
            config = yaml.safe_load(file)

        return config

    config = load_config("my_config.yaml")
    basicdata = pd.read_csv(path.join(config["data_directory"],
                            config["basic_data"]))
    CorpusPath = path.join(config["data_directory"], config["file_directory"])
    Corpusfiles = basicdata['Filename'].values

    # Segement size and overlapp offset value
    frameS = 256
    offset = 64

    spectrumCol = ['S'+str(i) for i in range(frameS)]
    spectrumData = np.empty((0, frameS), int)
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
                row = np.abs(np.fft.fft(segment))
                spectrumData = np.vstack([spectrumData, row])

    spectrumDF = pd.DataFrame(spectrumData, columns=spectrumCol)
    spectrumDF['Filename'] = Fcol

    # Combine basic data with features computed
    masterdata = pd.merge(basicdata, spectrumDF, on='Filename')
    masterdata.to_csv(path.join(config["data_directory"],
                                config["master_data"]), index=False)


if __name__ == '__main__':
    main()
