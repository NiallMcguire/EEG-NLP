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

if __name__ == "__main__":
    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"
    model_save_path = "/users/gxb18167/configs/model_checkpoints/"
    config_save_path = "/users/gxb18167/configs/"

    d = data.Data()
    util = utils.Utils()
    Loss = Loss
    Networks = Networks

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    parameters = {}
    epochs = 20
    parameters['epochs'] = epochs
    test_size = 0.2
    parameters['test_size'] = test_size
    validation_size = 0.2
    parameters['validation_size'] = validation_size
    num_negative_pairs_per_positive = 1
    parameters['num_negative_pairs_per_positive'] = num_negative_pairs_per_positive
    batch_size = 32
    parameters['batch_size'] = batch_size
    loss_function = "ContrastiveLossEuclidNER"
    parameters['loss_function'] = loss_function
    optimizer = "Adam"
    parameters['optimizer'] = optimizer
    learning_rate = 0.001
    parameters['learning_rate'] = learning_rate
    Embedding_model = 'BERT'  # 'Word2Vec' or 'BERT'
    parameters['Embedding_model'] = Embedding_model
    model_name = 'EEGToBERTModel_v2' # 'EEGToBERTModel_v1' or 'EEGToBERTModel_v2' or 'EEGToBERTModel_v3'
    parameters['model_name'] = model_name

    if Embedding_model == 'BERT':
        vector_size = 768
        parameters['vector_size'] = vector_size
        ner_bert = utils.NER_BERT()
        train_NE_embedded = ner_bert.get_embeddings(train_NE)

    elif Embedding_model == 'Word2Vec':
        vector_size = 50
        parameters['vector_size'] = vector_size
        window = 5
        parameters['window'] = window
        min_count = 1
        parameters['min_count'] = min_count
        workers = 4
        train_word_embeddings, train_NE_embedded = util.NER_Word2Vec(train_NE, vector_size, window, min_count, workers)


    train_NE_expanded = util.NER_expanded_NER_list(train_EEG_segments, train_NE_embedded, vector_size)
    train_NE_expanded = np.array(train_NE_expanded)

    X, y = util.NER_padding_x_y(train_EEG_segments, train_Classes)
    X = np.array(X)
    X = util.NER_reshape_data(X)
    y_categorical = util.encode_labels(y)

    # Create positive and negative pairs
    positive_pairs = [(X[i], train_NE_expanded[i], 1) for i in range(len(X))]
    negative_pairs = []

    # Create negative pairs
    for i in range(len(X)):
        negative_indices = sample([j for j in range(len(train_NE_expanded)) if j != i], num_negative_pairs_per_positive)
        for neg_index in negative_indices:
            negative_pairs.append((X[i], train_NE_expanded[neg_index], 0))

    # Combine positive and negative pairs
    all_pairs = positive_pairs + negative_pairs

    # Convert lists of numpy.ndarrays to single numpy.ndarrays
    eeg_array = np.array([pair[0] for pair in all_pairs])
    bert_array = np.array([pair[1] for pair in all_pairs])
    labels_array = np.array([pair[2] for pair in all_pairs])

    # Convert numpy.ndarrays to tensors
    eeg_pairs = torch.tensor(eeg_array, dtype=torch.float32)
    bert_pairs = torch.tensor(bert_array, dtype=torch.float32)
    labels = torch.tensor(labels_array, dtype=torch.float32)


    eeg_train, eeg_test, bert_train, bert_test, labels_train, labels_test = train_test_split(eeg_pairs, bert_pairs, labels, test_size=test_size, random_state=42)

    # validation
    eeg_train, eeg_val, bert_train, bert_val, labels_train, labels_val = train_test_split(eeg_train, bert_train, labels_train, test_size=validation_size, random_state=42)


    train_dataset = utils.EEGToBERTContrastiveDataset(eeg_train, bert_train, labels_train)
    validation_dataset = utils.EEGToBERTContrastiveDataset(eeg_val, bert_val, labels_val)

    test_dataset = utils.EEGToBERTContrastiveDataset(eeg_test, bert_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    eeg_input_dim = eeg_train.shape[2]  # Adjusted input dimension
    bert_output_dim = bert_train.shape[2]  # Keep as is since we have reshape the outputs


    # Assuming the model is already defined as EEGToBERTModel
    if model_name == 'EEGToBERTModel_v1':
        model = Networks.EEGToBERTModel_v1(eeg_input_dim, bert_output_dim)
    elif model_name == 'EEGToBERTModel_v2':
        model = Networks.EEGToBERTModel_v2(eeg_input_dim, bert_output_dim)
    elif model_name == 'EEGToBERTModel_v3':
        model = Networks.EEGToBERTModel_v3(eeg_input_dim, bert_output_dim)


    if loss_function == "ContrastiveLossEuclidNER":
        margin = 1.0
        parameters['margin'] = 1.0
        criterion = Loss.ContrastiveLossEuclidNER(margin=margin)

    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    patience = 5
    parameters['patience'] = patience

    def train_contrastive(model, train_loader, criterion, optimizer, num_epochs=epochs):
        model.train()
        best_validation_loss = float('inf')
        no_improvement_count = 0
        for epoch in range(num_epochs):
            running_loss = 0.0
            for eeg_vectors, bert_vectors, labels in train_loader:
                optimizer.zero_grad()
                output1 = model(eeg_vectors)
                output2 = bert_vectors  # Assuming bert_vectors are treated as target embeddings

                #print("labels shape", labels.shape)

                loss = criterion(output1, output2, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * eeg_vectors.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
            # Evaluate validation loss if validation loader is provided

            # Evaluate validation loss if validation loader is provided
            if validation_loader is not None:
                validation_loss = evaluate(model, validation_loader, criterion)
                print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {validation_loss:.4f}')

                # Check for early stopping criteria
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= patience:
                    print(f'No improvement for {patience} epochs. Stopping training.')
                    break

        return model


    def evaluate(model, data_loader, criterion) -> float:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for eeg_vectors, embeddings, labels in data_loader:
                output1 = model(eeg_vectors)
                output2 = embeddings
                loss = criterion(output1, output2, labels)
                total_loss += loss.item() * eeg_vectors.size(0)
                total_samples += eeg_vectors.size(0)
        return total_loss / total_samples

    model = train_contrastive(model, train_loader, criterion, optimizer)

    # model save path with the time stamp
    model_save_path = model_save_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "EEG_NER_Pre_Training.pt"
    torch.save(model.state_dict(), model_save_path)

    # Save the parameters
    parameters['model_save_path'] = model_save_path
    config_save_path = config_save_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "EEG_NER_Pre_Training.json"
    util.save_json(parameters, config_save_path)

    print("Model saved at: ", model_save_path)
    print("Config saved at: ", config_save_path)
    print("Training completed")
