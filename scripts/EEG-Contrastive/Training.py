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

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")


        EEG_X, named_entity_class = util.NER_padding_x_y(EEG_segments, Classes)
        EEG_X = np.array(EEG_X)
        EEG_X = util.NER_reshape_data(EEG_X)

        X, y = util.NER_padding_x_y(train_EEG_segments, train_Classes)
        X = np.array(X)
        X = util.NER_reshape_data(X)
        y_categorical = util.encode_labels(y)

        pre_train_model = Networks.SiameseNetwork_v2()

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

        # Convert numpy arrays to PyTorch tensors
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # Assuming your labels are integers

        x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)  # Assuming your labels are integers

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        train_size = len(train_dataset) - int(len(train_dataset) * 0.2)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                                   [train_size, int(len(train_dataset) * 0.2)])

        # Create the train loader
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

        pre_train_model.to(device)
        pre_train_model.eval()

        aligned_EEG = torch.empty((0, 64)).to(device)
        aligned_y = torch.empty((0, 3)).to(device)

        with torch.no_grad():
            for batch in train_loader:
                batch_EEG, batch_y = batch
                batch_EEG, batch_y = batch_EEG.to(device), batch_y.to(device)
                aligned_EEG_outputs = pre_train_model(batch_EEG, None)
                print(aligned_EEG_outputs.shape)
                aligned_EEG = torch.cat((aligned_EEG, aligned_EEG_outputs), dim=0)
                aligned_y = torch.cat((aligned_y, batch_y), dim=0)

        tensor_dataset = TensorDataset(aligned_EEG, aligned_y)

        print('Finished pre-training')
        print(' Tensors data shape:', aligned_EEG.shape, aligned_y.shape)


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
    pre_training_target_parameters['contrastive_learning_setting'] = ['EEGtoEEG']  # 'EEGtoBERT'
    pre_training_target_parameters['model_name'] = ['SiameseNetwork_v2']
    list_of_pre_trained_models, pre_trained_model_names, contrastive_learning_setting = util.find_target_models(
        config_save_path, pre_training_target_parameters)
    param_grid['pre_trained_model_path'] = list_of_pre_trained_models
    param_grid['pre_trained_model_name'] = pre_trained_model_names
    param_grid['contrastive_learning_setting'] = contrastive_learning_setting

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    for i in range(1):
        params = param_combinations[i]
        train_model = NER_Estimator(model_save_path, config_save_path, params)
        train_model.fit(train_NE, train_EEG_segments, train_Classes)