import json
import os



def fine_target_models(config_path, args):
    # print files in the directory
    pre_training_model_paths = []

    for file in os.listdir(config_path):
        # if file contains EEG_NER_Pre_Training and json file
        if 'EEG_NER_Pre_Training' in file and '.json' in file:
            #load json file
            with open(config_path + file, 'r') as doc:
                data = json.load(doc)
                # check if the file contains the target parameters
                for key in args.keys():
                    if key in data.keys():
                        for value in args[key]:
                            if value in data[key]:
                                # print model_save_path
                                print("Model matching target parameters saved @", data['model_save_path'])
                                pre_training_model_paths.append(data['model_save_path'])

    return pre_training_model_paths





if __name__ == "__main__":
    config_save_path = "/users/gxb18167/configs/"

    target_parameters = {}

    target_parameters['contrastive_learning_setting'] = ['EEGtoEEG', 'EEGtoBERT']
    target_parameters['model_name'] = ['SiameseNetwork_v1', 'SiameseNetwork_v2', 'SiameseNetwork_v3']


    # print files in the directory
    print("Files in the directory:")
    for file in os.listdir(config_save_path):
        # if file contains EEG_NER_Pre_Training and json file
        if 'EEG_NER_Pre_Training' in file and '.json' in file:
            #load json file
            with open(config_save_path + file, 'r') as doc:
                data = json.load(doc)
                # check if the file contains the target parameters
                for key in target_parameters.keys():
                    if key in data.keys():
                        for value in target_parameters[key]:
                            if value in data[key]:
                                # print model_save_path
                                print("Model matching target parameters saved @", data['model_save_path'])





