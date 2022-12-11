from os import path
import numpy as np
import pandas as pd
from utils.general import load_config

config = load_config("my_config.yaml")


class DataLoader:
    """
    A data loader class for DNN and CNN models
    """

    def __init__(self, mtype=config["default_model"]):
        self.mtype = mtype
        self.features = None
        self.labels = None

    def load_data(self):
        """
        A function to load dataset with respect to the model as 
        mentioned in the configuration settings
        """
        if self.mtype == 'myDNN':
            data = pd.read_csv(path.join(config["data_directory"],
                                         config["master_data"])).dropna()
            data = data.replace({'Speech': 0, 'Music': 1})
            self.features = data.drop(['Filename', 'Label'], axis=1).values
            self.labels = data.loc[:, ['Label']].values

        elif self.mtype == 'myCNN':
            self.features = np.load(path.join(config["data_directory"],
                                    config["cnn_X_data"]))
            self.labels = np.load(path.join(config["data_directory"],
                                  config["cnn_y_data"]))
