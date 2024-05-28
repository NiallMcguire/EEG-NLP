
import sys

from sklearn.model_selection import train_test_split

sys.path.append('/users/gxb18167/EEG-NLP')

from src import data
from src import utils
from src import Networks

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":

    parameters = {}


    evaluation = True
    pre_training = True

    parameters['evaluation'] = evaluation
    inputs = "EEG" # "EEG", "Text", "EEE+Text"
    parameters['inputs'] = inputs
    Embedding_model = 'BERT' # 'Word2Vec' or 'BERT'
    parameters['Embedding_model'] = Embedding_model
    batch_size = 32
    parameters['batch_size'] = batch_size
    input_size = 840
    parameters['input_size'] = input_size
    hidden_size = 64
    parameters['hidden_size'] = hidden_size
    dropout = 0.2
    parameters['dropout'] = dropout
    num_layers = 4
    parameters['num_layers'] = num_layers
    num_classes = 3
    parameters['num_classes'] = num_classes
    num_epochs = 100
    parameters['num_epochs'] = num_epochs
    LSTM_layers = 2
    parameters['LSTM_layers'] = LSTM_layers
    learning_rate = 0.001
    parameters['learning_rate'] = learning_rate
    optimizer = 'Adam'
    parameters['optimizer'] = optimizer
    criterion = 'CrossEntropyLoss'
    parameters['criterion'] = criterion
    val_size = 0.4
    parameters['val_size'] = val_size
    test_size = 0.9
    parameters['test_size'] = test_size

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"


    d = data.Data()
    util = utils.Utils()

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    if inputs == "EEE+Text" or "Text":
        #create word embeddings

        if Embedding_model == 'Word2Vec':
            vector_size = 50
            parameters['vector_size'] = vector_size
            window = 5
            parameters['window'] = window
            min_count = 1
            parameters['min_count'] = min_count
            workers = 4

            train_word_embeddings, train_NE_embedded = util.NER_Word2Vec(train_NE, vector_size, window, min_count, workers)

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

    #train test split
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
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, int(len(train_dataset) * val_size)])
    elif inputs == "Text":
        train_dataset = TensorDataset(train_NE_padded_tensor, y_train_tensor)
        test_dataset = TensorDataset(test_NE_padded_tensor, y_test_tensor)
        train_size = len(train_dataset) - int(len(train_dataset) * val_size)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, int(len(train_dataset) * val_size)])
    else:
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        train_size = len(train_dataset) - int(len(train_dataset) * val_size)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, int(len(train_dataset) * val_size)])

    # Create the train loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


    if pre_training == True:
        parameters['pre_training'] = pre_training
        pre_training_config = "/users/gxb18167/configs/20240528-130339EEG_NER_Pre_Training.json"
        parameters['pre_training_config'] = pre_training_config

        # load pre-training config
        pre_training_config = util.load_json(pre_training_config)
        model_save_path = pre_training_config['model_save_path']

        # load pre-trained model

        pre_train_model = Networks.EEGToBERTModel(input_size, vector_size)
        pre_train_model.load_state_dict(torch.load(model_save_path))

        pre_train_model.to(device)
        pre_train_model.eval()
        #replace train_loader with new encoded data
        # Initialize empty tensors
        train_aligned_EEG = torch.empty((0, vector_size)).to(device)
        train_aligned_y = torch.empty((0,)).to(device)

        validation_aligned_EEG = torch.empty((0, vector_size)).to(device)
        validation_aligned_y = torch.empty((0,)).to(device)

        test_aligned_EEG = torch.empty((0, vector_size)).to(device)
        test_aligned_y = torch.empty((0,)).to(device)

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


        # Create the train loader
        train_aligned_loader = DataLoader(dataset=train_aligned_EEG, batch_size=batch_size, shuffle=True)
        validation_aligned_loader = DataLoader(dataset=validation_aligned_EEG, batch_size=batch_size, shuffle=False)
        test_aligned_loader = DataLoader(dataset=test_aligned_EEG, batch_size=batch_size, shuffle=False)

    print("Pre-training complete")

    '''
    # Instantiate the model
    if inputs == "EEG+Text":
        model = Networks.BLSTM_Text(input_size, vector_size, hidden_size, num_classes, num_layers, dropout)
    elif inputs == "Text":
        input_size = vector_size
        parameters['input_size'] = input_size
        model = Networks.BLSTM(input_size, hidden_size, num_classes, num_layers, dropout)
    else:
        model = Networks.BLSTM(input_size, hidden_size, num_classes,num_layers, dropout)
    model.to(device)

    # Define loss function and optimizer
    if criterion == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()

    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #early stopping
    patience = 4
    parameters['patience'] = patience
    counter = 0
    best_val_loss = None
    best_model = None

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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        #early stopping
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
            print(f'Validation loss: {val_loss:.4f}')
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
    '''

    '''
    # Save the trained model
    torch.save(model.state_dict(), 'blstm_model.pth')

    # Load the saved model
    loaded_model = Networks.BLSTM(input_size, hidden_size, num_layers, num_classes, LSTM_layers)
    loaded_model.load_state_dict(torch.load('blstm_model.pth'))
    loaded_model.eval()  # Switch to evaluation mode
    '''

    '''
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
    '''






