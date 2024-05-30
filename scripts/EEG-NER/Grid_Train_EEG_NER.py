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

class NER_Estimator():
    def __init__(self, model_save_path, config_save_path, kwargs):
        self.parameters = kwargs
        self.model_save_path = model_save_path
        self.config_save_path = config_save_path
        self.model = None
        self.device = None

    def fit(self, train_NE, train_EEG_segments, train_Classes):

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.device = device

        # Define parameters from kwargs
        inputs = self.parameters['inputs']
        Embedding_model = self.parameters['Embedding_model']
        batch_size = self.parameters['batch_size']
        input_size = self.parameters['input_size']
        hidden_size = self.parameters['hidden_size']
        dropout = self.parameters['dropout']
        num_layers = self.parameters['num_layers']
        num_classes = self.parameters['num_classes']
        num_epochs = self.parameters['num_epochs']
        LSTM_layers = self.parameters['LSTM_layers']
        learning_rate = self.parameters['learning_rate']
        optimizer = self.parameters['optimizer']
        criterion = self.parameters['criterion']
        val_size = self.parameters['val_size']
        test_size = self.parameters['test_size']
        pre_training = self.parameters['pre_training']
        evaluation = self.parameters['evaluation']
        patience = self.parameters['Patience']
        cross_val = self.parameters['cross_val']
        parameters = self.parameters

        # cross validation
        cross_val_accuracy = []
        for i in range(cross_val):

            if inputs == "EEE+Text" or "Text":
                # create word embeddings

                if Embedding_model == 'Word2Vec':
                    vector_size = 50
                    parameters['vector_size'] = vector_size
                    window = 5
                    parameters['window'] = window
                    min_count = 1
                    parameters['min_count'] = min_count
                    workers = 4

                    train_word_embeddings, train_NE_embedded = util.NER_Word2Vec(train_NE, vector_size, window, min_count,
                                                                                 workers)

                elif Embedding_model == 'BERT':
                    vector_size = 768
                    parameters['vector_size'] = vector_size

                    ner_bert = utils.NER_BERT()

                    train_NE_embedded = ner_bert.get_embeddings(train_NE)

                train_NE_expanded = util.NER_expanded_NER_list(train_EEG_segments, train_NE_embedded, vector_size)

                train_NE_expanded = np.array(train_NE_expanded)

                train_NE_padded_tensor = torch.tensor(train_NE_expanded, dtype=torch.float32)

            X, y = util.NER_padding_x_y(train_EEG_segments, train_Classes)
            X = np.array(X)
            X = util.NER_reshape_data(X)
            y_categorical = util.encode_labels(y)

            train_NE_padded_tensor, test_NE_padded_tensor, _, _ = train_test_split(
                train_NE_padded_tensor, y_categorical, test_size=test_size, random_state=42)

            # train test split
            X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, random_state=42)

            # Convert numpy arrays to PyTorch tensors
            x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # Assuming your labels are integers

            x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)  # Assuming your labels are integers

            if inputs == "EEG+Text":
                train_dataset = TensorDataset(x_train_tensor, train_NE_padded_tensor, y_train_tensor)
                test_dataset = TensorDataset(x_test_tensor, test_NE_padded_tensor, y_test_tensor)

                train_size = len(train_dataset) - int(len(train_dataset) * val_size)
                train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                                           [train_size, int(len(train_dataset) * val_size)])
            elif inputs == "Text":
                train_dataset = TensorDataset(train_NE_padded_tensor, y_train_tensor)
                test_dataset = TensorDataset(test_NE_padded_tensor, y_test_tensor)
                train_size = len(train_dataset) - int(len(train_dataset) * val_size)
                train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                                           [train_size, int(len(train_dataset) * val_size)])
            else:
                train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
                test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

                train_size = len(train_dataset) - int(len(train_dataset) * val_size)
                train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                                           [train_size, int(len(train_dataset) * val_size)])

            # Create the train loader
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

            print("Data before pre-training: ", train_dataset[0][0].shape)

            if pre_training == True:
                parameters['pre_training'] = pre_training

                # load pre-training config
                model_save_path = parameters['pre_trained_model_path']

                # load pre-trained model
                if parameters['pre_trained_model_name'] == 'EEGToBERTModel_v4':
                    pre_train_model = Networks.EEGToBERTModel_v4(input_size, vector_size)
                elif parameters['pre_trained_model_name'] == 'EEGToBERTModel_v3':
                    pre_train_model = Networks.EEGToBERTModel_v3(input_size, vector_size)

                pre_train_model.load_state_dict(torch.load(model_save_path))

                pre_train_model.to(device)
                pre_train_model.eval()
                # replace train_loader with new encoded data
                # Initialize empty tensors
                train_aligned_EEG = torch.empty((0, 7, 768)).to(device)  # Initialize with the correct shape
                train_aligned_y = torch.empty((0, 3)).to(device)

                validation_aligned_EEG = torch.empty((0, 7, 768)).to(device)
                validation_aligned_y = torch.empty((0, 3)).to(device)

                test_aligned_EEG = torch.empty((0, 7, 768)).to(device)
                test_aligned_y = torch.empty((0, 3)).to(device)

                with torch.no_grad():
                    for batch in train_loader:
                        batch_EEG, batch_y = batch
                        batch_EEG, batch_y = batch_EEG.to(device), batch_y.to(device)
                        aligned_EEG_outputs = pre_train_model(batch_EEG)
                        train_aligned_EEG = torch.cat((train_aligned_EEG, aligned_EEG_outputs), dim=0)
                        train_aligned_y = torch.cat((train_aligned_y, batch_y), dim=0)

                    for batch in val_loader:
                        batch_EEG, batch_y = batch
                        batch_EEG, batch_y = batch_EEG.to(device), batch_y.to(device)
                        aligned_EEG_outputs = pre_train_model(batch_EEG)
                        validation_aligned_EEG = torch.cat((validation_aligned_EEG, aligned_EEG_outputs), dim=0)
                        validation_aligned_y = torch.cat((validation_aligned_y, batch_y), dim=0)

                    for batch in test_loader:
                        batch_EEG, batch_y = batch
                        batch_EEG, batch_y = batch_EEG.to(device), batch_y.to(device)
                        aligned_EEG_outputs = pre_train_model(batch_EEG)
                        test_aligned_EEG = torch.cat((test_aligned_EEG, aligned_EEG_outputs), dim=0)
                        test_aligned_y = torch.cat((test_aligned_y, batch_y), dim=0)

                # Create TensorDataset instances
                train_dataset = TensorDataset(train_aligned_EEG, train_aligned_y)
                validation_dataset = TensorDataset(validation_aligned_EEG, validation_aligned_y)
                test_dataset = TensorDataset(test_aligned_EEG, test_aligned_y)

                # Re-create the data loaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                print("Pre-training complete")
                print("train aligned shape: ", train_aligned_EEG.shape)
                input_size = 768

            # Instantiate the model
            if inputs == "EEG+Text":
                model = Networks.BLSTM_Text(input_size, vector_size, hidden_size, num_classes, num_layers, dropout)
            elif inputs == "Text":
                input_size = vector_size
                parameters['input_size'] = input_size
                model = Networks.BLSTM(input_size, hidden_size, num_classes, num_layers, dropout)
            else:
                model = Networks.BLSTM(input_size, hidden_size, num_classes, num_layers, dropout)

            # Move the model to the GPU if available
            model.to(device)

            # Define loss function and optimizer
            if criterion == 'CrossEntropyLoss':
                criterion = nn.CrossEntropyLoss()

            if optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # early stopping
            counter = 0
            best_val_loss = None

            loss_over_batches = []
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for batch in train_loader:
                    if inputs == "EEG+Text":
                        batch_x, batch_NE, batch_y = batch
                        batch_x, batch_NE, batch_y = batch_x.to(device), batch_NE.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_x, batch_NE)
                    else:
                        batch_x, batch_y = batch
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_x)

                    # Convert class probabilities to class indices
                    _, predicted = torch.max(outputs, 1)

                    loss = criterion(outputs, batch_y.squeeze())  # Ensure target tensor is Long type
                    loss_over_batches.append(loss.item())

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

                # early stopping
                model.eval()
                with torch.no_grad():

                    val_loss = 0
                    for batch in val_loader:
                        if inputs == "EEG+Text":
                            batch_x, batch_NE, batch_y = batch
                            batch_x, batch_NE, batch_y = batch_x.to(device), batch_NE.to(device), batch_y.to(device)
                            outputs = model(batch_x, batch_NE)
                        else:
                            batch_x, batch_y = batch
                            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                            outputs = model(batch_x)
                        loss = criterion(outputs, batch_y.squeeze())
                        val_loss += loss.item()
                    #print(f'Validation loss: {val_loss:.4f}')
                    if best_val_loss is None:
                        best_val_loss = val_loss
                        best_model = model.state_dict()
                    elif val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model.state_dict()
                        counter = 0
                    else:
                        counter += 1
                        if counter > patience:
                            print("Early stopping")
                            break

            parameters['Loss'] = loss_over_batches

            if evaluation == True:
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for batch in test_loader:
                        if inputs == "EEG+Text":
                            batch_x, batch_NE, batch_y = batch
                            batch_x, batch_NE, batch_y = batch_x.to(device), batch_NE.to(device), batch_y.to(device)
                            outputs = model(batch_x, batch_NE)
                        else:
                            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                            outputs = model(batch_x)
                        _, predicted = torch.max(outputs, 1)
                        total += batch_y.size(0)
                        correct += (predicted == torch.argmax(batch_y, 1)).sum().item()
                    print('Accuracy of the model on the test set: {}%'.format(100 * correct / total))
                    parameters['Accuracy'] = 100 * correct / total
                    cross_val_accuracy.append(100 * correct / total)


        print("Mean accuracy: ", np.mean(cross_val_accuracy))
        parameters['Mean_Accuracy'] = np.mean(cross_val_accuracy)

        model_save_path = self.model_save_path + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S") + "EEG_NER.pt"
        torch.save(model.state_dict(), model_save_path)

        # Save the parameters
        parameters['model_save_path'] = model_save_path
        config_save_path = self.config_save_path + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S") + "EEG_NER.json"
        util.save_json(parameters, config_save_path)

        print("Model saved at: ", model_save_path)
        print("Config saved at: ", config_save_path)
        print("Training completed")



if __name__ == "__main__":
    # Define parameter grid for grid search

    '''
    param_grid = {
        'epochs': [100],
        'patience': [5, 10],
        'test_size': [0.2],
        'validation_size': [0.2],
        'num_negative_pairs_per_positive': [1, 2],
        'batch_size': [32, 64, 128],
        'loss_function': ["ContrastiveLossEuclidNER"],
        'margin': [0.5, 1.0, 2.0],
        'optimizer': ["Adam"],
        'learning_rate': [0.001, 0.0001],
        'Embedding_model': ['BERT'],
        'model_name': ['EEGToBERTModel_v4', 'EEGToBERTModel_v3']
    }
    '''
    param_grid = {
        'pre_training': [True],
        'evaluation': [True],
        'Patience': [10],
        'inputs': ["EEG+Text"], # EEG, Text, EEG+Text
        'Embedding_model': ['BERT'],
        'batch_size': [32],
        'input_size': [840],
        'hidden_size': [64],
        'dropout': [0.2],
        'num_layers': [4],
        'num_classes': [3],
        'num_epochs': [100],
        'LSTM_layers': [2],
        'learning_rate': [0.001],
        'optimizer': ['Adam'],
        'criterion': ['CrossEntropyLoss'],
        'val_size': [0.1],
        'test_size': [0.1],
        'cross_val': [3]
    }

    models = ['EEGToBERTModel_v4', 'EEGToBERTModel_v3']

    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"
    model_save_path = "/users/gxb18167/configs/model_checkpoints/"
    config_save_path = "/users/gxb18167/configs/"

    d = data.Data()
    util = utils.Utils()
    Loss = Loss
    Networks = Networks

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    if param_grid['pre_training'] == [True]:
        model_save_paths, model_names = util.load_pre_training_gridsearch(models, config_save_path)
        #for pre_trained_models_path in model_save_paths:
        for i in range(0, 1):
            pre_trained_models_path = model_save_paths[i]
            model_name = model_names[i]
            for params in param_combinations:
                params['pre_trained_model_path'] = pre_trained_models_path
                params['pre_trained_model_name'] = model_name
                train_model = NER_Estimator(model_save_path, config_save_path, params)
                train_model.fit(train_NE, train_EEG_segments, train_Classes)
                #print("Model trained with parameters: ", params)

    else:
        for params in param_combinations:
            train_model = NER_Estimator(model_save_path, config_save_path, params)
            train_model.fit(train_NE, train_EEG_segments, train_Classes)
            # print("Model trained with parameters: ", params)


