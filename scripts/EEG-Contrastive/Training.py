import itertools
import sys

from sklearn.model_selection import train_test_split

sys.path.append('/users/gxb18167/EEG-NLP')
from src import data
from src import utils
from src import Networks
from src import Loss

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import datetime

class NER_Estimator:
    def __init__(self, model_save_path, config_save_path, params):
        self.model_save_path = model_save_path
        self.config_save_path = config_save_path
        self.params = params

    def fit(self, NE, EEG_segments, Classes):

        ner_bert = utils.NER_BERT()

        EEG_X, named_entity_class = util.NER_padding_x_y(EEG_segments, Classes)
        EEG_X = np.array(EEG_X)
        EEG_X = util.NER_reshape_data(EEG_X)

        pairs, labels = d.NER_EEGtoEEG_create_paris(EEG_X, named_entity_class, max_positive_pairs, max_negative_pairs)
        print("Created EEG to EEG pairs of shape: ", pairs.shape)
        pair_one = pairs[:, 0]
        pair_two = pairs[:, 1]



if __name__ == "__main__":
    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"
    model_save_path = "/users/gxb18167/configs/model_checkpoints/"
    config_save_path = "/users/gxb18167/configs/"

    d = data.Data()
    util = utils.Utils()
    Loss = Loss
    Networks = Networks

    param_grid = {}

    pre_training_target_parameters = {}
    pre_training_target_parameters['contrastive_learning_setting'] = ['EEGtoBERT']  # 'EEGtoBERT'
    pre_training_target_parameters['model_name'] = ['SiameseNetwork_v1', 'SiameseNetwork_v2', 'SiameseNetwork_v3']
    list_of_pre_trained_models, pre_trained_model_names, contrastive_learning_setting = util.find_target_models(
        config_save_path, pre_training_target_parameters)
    param_grid['pre_trained_model_path'] = list_of_pre_trained_models
    param_grid['pre_trained_model_name'] = pre_trained_model_names
    param_grid['contrastive_learning_setting'] = contrastive_learning_setting

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]



    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    for params in param_combinations:
        train_model = NER_Estimator(model_save_path, config_save_path, params)
        train_model.fit(train_NE, train_EEG_segments, train_Classes)