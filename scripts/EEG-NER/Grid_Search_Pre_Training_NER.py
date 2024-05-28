import sys
sys.path.append('/users/gxb18167/EEG-NLP')
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator
import torch.nn.functional as F
from src import data
from src import utils
from src import Networks
from src import Loss
import numpy as np
from random import sample
import datetime
from torch.utils.data import Dataset, DataLoader

class EEGToBERTModelEstimator(BaseEstimator):
    def __init__(self, model_save_path, config_save_path, **kwargs):
        self.parameters = kwargs
        self.model_save_path = model_save_path
        self.config_save_path = config_save_path

    def fit(self, train_NE,train_EEG_segments, train_Classes):
        parameters = self.parameters


if __name__ == "__main__":
    # Define parameter grid for grid search
    param_grid = {
        'epochs': [100, 200],
        'patience': [5, 10],
        'test_size': [0.2, 0.3],
        'validation_size': [0.1, 0.2],
        'num_negative_pairs_per_positive': [1, 2],
        'batch_size': [32, 64],
        'loss_function': ["ContrastiveLossEuclidNER"],
        'margin': [1.0, 2.0],
        'optimizer': ["Adam", "SGD"],
        'learning_rate': [0.001, 0.01],
        'Embedding_model': ['BERT', 'Word2Vec'],
        'model_name': ['EEGToBERTModel_v1', 'EEGToBERTModel_v2', 'EEGToBERTModel_v3']
    }

    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"
    model_save_path = "/users/gxb18167/configs/model_checkpoints/"
    config_save_path = "/users/gxb18167/configs/"

    d = data.Data()
    util = utils.Utils()
    Loss = Loss
    Networks = Networks

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    X = (train_NE, train_EEG_segments)
    y = train_Classes

    train_model = EEGToBERTModelEstimator(model_save_path, config_save_path)


    grid_search = GridSearchCV(estimator=train_model, param_grid=param_grid, cv=3)
    grid_search.fit(X, y)

    print("Best parameters found: ", grid_search.best_params_)
