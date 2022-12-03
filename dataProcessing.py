from os import path
import pandas as pd
from utils.general import load_config, generateDNNData, generateCNNData


def main():

    config = load_config("my_config.yaml")
    basicdata = pd.read_csv(path.join(config["data_directory"],
                            config["basic_data"]))

    # Segement size and overlapp offset value
    frameS = 256
    offset = 64
    generateCNNData(basicdata, frameS=frameS, offset=offset)
    generateDNNData(basicdata, frameS=frameS, offset=offset)


if __name__ == '__main__':
    main()
