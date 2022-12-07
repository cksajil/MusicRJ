from os import path
import pandas as pd
from utils.general import load_config, generateDNNData, generateCNNData


def main():

    config = load_config("my_config.yaml")
    basicdata = pd.read_csv(path.join(config["data_directory"],
                            config["basic_data"]))

    if config['default_model'] == 'myDNN':
        generateDNNData(basicdata,
                        frameS=config['FRAMES'],
                        offset=config['OFFSET'])
    else:
        generateCNNData(basicdata,
                        frameS=config['FRAMES'],
                        offset=config['OFFSET'])


if __name__ == '__main__':
    main()
