
import sys
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
    log = {}

    parameters = {}

    evaluation = True
    parameters['evaluation'] = evaluation
    EEG_with_Text = True
    parameters['EEG_with_Text'] = EEG_with_Text
    Embedding_model = 'Word2Vec'
    parameters['Embedding_model'] = Embedding_model

    batch_size = 32
    parameters['batch_size'] = batch_size
    input_size = 840
    parameters['input_size'] = input_size
    hidden_size = 64
    parameters['hidden_size'] = hidden_size
    num_layers = 2
    parameters['num_layers'] = num_layers
    num_classes = 3
    parameters['num_classes'] = num_classes
    num_epochs = 10
    parameters['num_epochs'] = num_epochs
    LSTM_layers = 2
    parameters['LSTM_layers'] = LSTM_layers
    learning_rate = 0.001

    parameters['learning_rate'] = learning_rate
    optimizer = 'Adam'
    parameters['optimizer'] = optimizer
    criterion = 'CrossEntropyLoss'
    parameters['criterion'] = criterion

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_path = r"/users/gxb18167/Datasets/ZuCo/train_NER.pkl"

    test_path = r"/users/gxb18167/Datasets/ZuCo/test_NER.pkl"


    d = data.Data()
    util = utils.Utils()

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)
    test_NE, test_EEG_segments, test_Classes = d.NER_read_custom_files(test_path)

    if EEG_with_Text == True:
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
            test_word_embeddings, test_NE_embedded = util.NER_Word2Vec(test_NE, vector_size, window, min_count, workers)

            assert len(train_NE_embedded) == len(train_Classes)


            #paddding
            train_NE_padded, y = util.NER_padding_x_y(train_NE_embedded, train_Classes, vector_size)
            test_NE_padded, _ = util.NER_padding_x_y(test_NE_embedded, test_Classes, vector_size)

            print("type of train_NE_padded", type(train_NE_padded))
            print("Length of train_NE_padded", len(train_NE_padded))
            print("lenght of y ", len(y))

            print(train_NE_padded.shape)

            train_NE_padded = np.array(train_NE_padded)
            test_NE_padded = np.array(test_NE_padded)


            train_NE_padded_tensor = torch.tensor(train_NE_padded, dtype=torch.float32)
            test_NE_padded_tensor = torch.tensor(test_NE_padded, dtype=torch.float32)

            breakpoint()






    # padding
    X_train, y_train = util.NER_padding_x_y(train_EEG_segments, train_Classes)
    X_train_numpy = np.array(X_train)
    X_train_numpy = util.NER_reshape_data(X_train_numpy)
    y_train_categorical = util.encode_labels(y_train)

    X_test, y_test = util.NER_padding_x_y(test_EEG_segments, test_Classes)
    X_test_numpy = np.array(X_test)
    X_test_numpy = util.NER_reshape_data(X_test_numpy)
    y_test_categorical = util.encode_labels(y_test)

    # Convert numpy arrays to PyTorch tensors
    print(X_train_numpy.shape)
    x_train_tensor = torch.tensor(X_train_numpy, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_categorical, dtype=torch.float32)  # Assuming your labels are integers

    x_test_tensor = torch.tensor(X_test_numpy, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_categorical, dtype=torch.float32)  # Assuming your labels are integers


    if EEG_with_Text == True:
        train_dataset = TensorDataset(x_train_tensor, train_NE_padded_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, test_NE_padded_tensor, y_test_tensor)
    else:
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create the train loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    model = Networks.BLSTM(input_size, hidden_size, num_layers, num_classes, LSTM_layers)
    model.to(device)

    # Define loss function and optimizer
    if criterion == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()

    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_over_batches = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
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

    log['Loss'] = loss_over_batches
    '''
    # Save the trained model
    torch.save(model.state_dict(), 'blstm_model.pth')

    # Load the saved model
    loaded_model = Networks.BLSTM(input_size, hidden_size, num_layers, num_classes, LSTM_layers)
    loaded_model.load_state_dict(torch.load('blstm_model.pth'))
    loaded_model.eval()  # Switch to evaluation mode
    '''

    if evaluation == True:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == torch.argmax(batch_y, 1)).sum().item()
            print('Accuracy of the model on the test set: {}%'.format(100 * correct / total))
            log['Accuracy'] = 100 * correct / total






