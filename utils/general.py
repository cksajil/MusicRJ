import yaml
import numpy as np
from os import path
from utils.dnn import myDNN


def load_config(config_name):
    CONFIG_PATH = "./config/"
    with open(path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


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
    config = load_config("my_config.yaml")
    if config['default_model'] == 'myDNN':
        model = myDNN(in_shape)
    else:
        pass
    return model
