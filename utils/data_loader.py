from os import path
import numpy as np
import pandas as pd
from utils.general import load_config

config = load_config("my_config.yaml")


class DataLoader:

    def __init__(self, mtype='one_dim'):
        self.data_dim = mtype
        self.features = None
        self.labels = None

    def load_data(self):
        if self.data_dim == 'one_dim':
            data = pd.read_csv(path.join(config["data_directory"],
                                         config["master_data"])).dropna()
            data = data.replace({'Speech': 0, 'Music': 1})
            self.features = data.drop(['Filename', 'Label'], axis=1).values
            self.labels = data.loc[:, ['Label']].values

        elif self.data_dim == 'two_dim':
            self.features = np.load(path.join(config["data_directory"],
                                    config["cnn_X_data"]))
            self.labels = np.load(path.join(config["data_directory"],
                                  config["cnn_y_data"]))
