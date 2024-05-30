import sys
import os

sys.path.append('/users/gxb18167/EEG-NLP')

from src import utils


if __name__ == "__main__":

    util = utils.Utils()

    config_paths = "/users/gxb18167/configs/"


    #list all files in directory
    for file in os.listdir(config_paths):
        #check if file is a .json file and contains NER
        if file.endswith(".json") and "NER" in file:
            parameter_dictionary = util.load_json(config_paths + file)

