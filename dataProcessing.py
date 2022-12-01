from os import path
import pandas as pd
from utils.general import load_config, generateDNNData


def main():

    config = load_config("my_config.yaml")
    basicdata = pd.read_csv(path.join(config["data_directory"],
                            config["basic_data"]))

    # Segement size and overlapp offset value
    frameS = 256
    offset = 64
    spectrumDF = generateDNNData(basicdata, frameS=frameS, offset=offset)

    # Combine basic data with features computed
    masterdata = pd.merge(basicdata, spectrumDF, on='Filename')
    masterdata.to_csv(path.join(config["data_directory"],
                                config["master_data"]), index=False)


if __name__ == '__main__':
    main()
