import sys

sys.path.append('/users/gxb18167/EEG-NLP')


import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from src import data
from src import utils
from src import Networks
from src import Loss
import numpy as np
from random import sample
import datetime
from torch.utils.data import Dataset, DataLoader


class PreTraining():
    def __init__(self, Data, model_save_path, config_save_path, kwargs):
        self.parameters = kwargs
        self.data = Data
        self.model_save_path = model_save_path
        self.config_save_path = config_save_path
        self.model = None
        self.device = None

    def train(self):
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device


        NE, EEG_segments, Classes = self.data

        max_positive_pairs = self.parameters['max_positive_pairs']
        max_negative_pairs = self.parameters['max_negative_pairs']
        contrastive_learning_setting = self.parameters['contrastive_learning_setting']


        epochs = self.parameters['epochs']
        batch_size = self.parameters['batch_size']
        test_size = self.parameters['test_size']
        validation_size = self.parameters['validation_size']

        loss_function = self.parameters['loss_function']
        margin = self.parameters['margin']
        optimizer = self.parameters['optimizer']
        learning_rate = self.parameters['learning_rate']
        model_name = self.parameters['model_name']
        patience = self.parameters['patience']


        ner_bert = utils.NER_BERT()

        EEG_X, named_entity_class = util.NER_padding_x_y(EEG_segments, Classes)
        EEG_X = np.array(EEG_X)
        EEG_X = util.NER_reshape_data(EEG_X)

        if contrastive_learning_setting == "EEGtoBERT":
            NE_embedded = ner_bert.get_embeddings(NE)
            NE_expanded = util.NER_expanded_NER_list(EEG_segments, NE_embedded, 768)
            NE_expanded = np.array(NE_expanded)
            pair_one, pair_two, labels = d.NER_EEGtoBERT_create_pairs(EEG_X, NE_expanded, named_entity_class,
                                                                    max_positive_pairs, max_negative_pairs)
        elif contrastive_learning_setting == "EEGtoEEG":
            pairs, labels = d.NER_EEGtoEEG_create_paris(EEG_X, named_entity_class, max_positive_pairs, max_negative_pairs)
            print("Created EEG to EEG pairs of shape: ", pairs.shape)
            pair_one = pairs[:, 0]
            pair_two = pairs[:, 1]

        # Convert to tensors
        labels = torch.tensor(labels, dtype=torch.float32)
        pair_one = torch.tensor(pair_one, dtype=torch.float32)
        pair_two = torch.tensor(pair_two, dtype=torch.float32)

        # Split data into training and testing sets using tensors
        pair_one_train, pair_one_test, pair_two_train, pair_two_test, labels_train, labels_test = train_test_split(
            pair_one, pair_two, labels, test_size=test_size, random_state=42)

        # Split training data into training and validation sets using tensors
        pair_one_train, pair_one_val, pair_two_train, pair_two_val, labels_train, labels_val = train_test_split(
            pair_one_train, pair_two_train, labels_train, test_size=validation_size, random_state=42)

        # Create datasets
        train_dataset = utils.EEGContrastiveDataset(pair_one_train, pair_two_train, labels_train)
        val_dataset = utils.EEGContrastiveDataset(pair_one_val, pair_two_val, labels_val)
        test_dataset = utils.EEGContrastiveDataset(pair_one_test, pair_two_test, labels_test)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Print shapes
        print("Training data shapes: ", pair_one_train.shape, pair_two_train.shape, labels_train.shape)

        eeg_input_dim = pair_one_train.shape[2]
        print("EEG input dimension: ", eeg_input_dim)


        # Initialize loss function
        if loss_function == "ContrastiveLossEuclidNER":
            criterion = Loss.ContrastiveLossEuclidNER(margin=margin)



        # Initialize model
        if model_name == "SiameseNetwork_v1":
            model = Networks.SiameseNetwork_v1(7*840, 7*768).to(device)
        elif model_name == "SiameseNetwork_v2":
            model = Networks.SiameseNetwork_v2().to(device)
        elif model_name == "SiameseNetwork_v3":
            model = Networks.SiameseNetwork_v3(840, 768).to(device)

        if optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        num_epochs = epochs
        # early stopping
        best_val_loss = float('inf')
        counter = 0
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                pair_one_batch, pair_two_batch, label_batch = batch
                pair_one_batch, pair_two_batch, label_batch = pair_one_batch.to(device), pair_two_batch.to(
                    device), label_batch.to(device)

                optimizer.zero_grad()
                output1, output2 = model(pair_one_batch, pair_two_batch)
                loss = criterion(output1, output2, label_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

            # validation with early stopping
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    pair_one_batch, pair_two_batch, label_batch = batch
                    pair_one_batch, pair_two_batch, label_batch = pair_one_batch.to(device), pair_two_batch.to(
                        device), label_batch.to(device)

                    output1, output2 = model(pair_one_batch, pair_two_batch)
                    loss = criterion(output1, output2, label_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break




if __name__ == "__main__":
    #data_path = r"C:\Users\gxb18167\PycharmProjects\EEG-NLP\NER.pkl"
    data_path = r"/users/gxb18167/EEG-NLP/NER.pkl"


    param_grid = {
        'epochs': [100],
        'patience': [5],
        'test_size': [0.2],
        'validation_size': [0.1],
        'max_positive_pairs': [20000],
        'max_negative_pairs': [20000],
        'contrastive_learning_setting': ['EEGtoBERT'],  # 'EEGtoBERT', 'EEGtoEEG
        'batch_size': [32],
        'loss_function': ["ContrastiveLossEuclidNER"],
        'margin': [0.5],
        'optimizer': ["Adam"],
        'learning_rate': [0.0001],
        'model_name': ['SiameseNetwork_v1']
    }

    d = data.Data()
    util = utils.Utils()
    Loss = Loss
    Networks = Networks

    NE, EEG_segments, Classes = d.NER_read_custom_files(data_path)

    PreTraining = PreTraining((NE, EEG_segments, Classes), model_save_path=None, config_save_path=None, kwargs=param_grid)
    PreTraining.train()



















