import sys
import os

sys.path.append('/users/gxb18167/EEG-NLP')

from src import utils


if __name__ == "__main__":

    util = utils.Utils()

    config_paths = "/users/gxb18167/configs/"

    models = ['EEGToBERTModel_v4', 'EEGToBERTModel_v3']


    #list all files in directory
    for file in os.listdir(config_paths):
        #check if file is a .json file and contains NER
        if file.endswith(".json") and "NER_Pre_Training" in file:
            parameter_dictionary = util.load_json(config_paths + file)
            if 'model_name' in parameter_dictionary:
                for model in models:
                    if model in parameter_dictionary['model_name']:
                        print(f"Loading model: {model}")



