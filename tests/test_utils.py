from utils.general import selectModel


def test_selectModel():
    assert type(selectModel()) == str
