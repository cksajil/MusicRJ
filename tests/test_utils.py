from utils.general import load_config, selectModel


def test_selectModel():
    assert type(selectModel()) == str


def test_load_config():
    assert type(load_config("my_config.yaml")) == dict
