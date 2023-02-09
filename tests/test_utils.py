from utils.general import load_config, selectModel
from utils.general import strided_app, create_model
import numpy as np


def test_selectModel():
    assert type(selectModel()) == str


def test_load_config():
    assert type(load_config("my_config.yaml")) == dict


def test_strided_app():
    a = np.array((range(100)))
    L = 20
    S = 3
    assert strided_app(a, L, S).shape == (27, 20)


def test_create_model():
    assert (create_model((256,)).output_shape) == (None, 2)
