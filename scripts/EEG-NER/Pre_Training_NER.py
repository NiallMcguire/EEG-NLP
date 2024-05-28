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


from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"
    #save_path = r"/users/gxb18167/EEG-NLP/"

    d = data.Data()
    util = utils.Utils()
    Loss = Loss
    Networks = Networks

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    parameters = {}
    test_size = 0.2
    parameters['test_size'] = test_size
    num_negative_pairs_per_positive = 1
    parameters['num_negative_pairs_per_positive'] = num_negative_pairs_per_positive
    batch_size = 32
    parameters['batch_size'] = batch_size

    Embedding_model = 'BERT'  # 'Word2Vec' or 'BERT'
    parameters['Embedding_model'] = Embedding_model

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
    eeg_train, eeg_val, bert_train, bert_val, labels_train, labels_val = train_test_split(eeg_train, bert_train, labels_train, test_size=0.2, random_state=42)


    train_dataset = utils.EEGToBERTContrastiveDataset(eeg_train, bert_train, labels_train)
    validation_dataset = utils.EEGToBERTContrastiveDataset(eeg_val, bert_val, labels_val)

    test_dataset = utils.EEGToBERTContrastiveDataset(eeg_test, bert_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    print("EEG input shape: ", eeg_train.shape)
    print("BERT input shape: ", bert_train.shape)

    print("EEG input shape 1", eeg_train.shape[1])
    print("BERT input shape 1", bert_train.shape[1])


    eeg_input_dim = eeg_train.shape[2]  # Adjusted input dimension
    bert_output_dim = bert_train.shape[2]  # Keep as is since we have reshape the outpus


    # Assuming the model is already defined as EEGToBERTModel
    model = Networks.EEGToBERTModel(eeg_input_dim, bert_output_dim)
    criterion = Loss.ContrastiveLossEuclidNER(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    patience = 5


    def train_contrastive(model, train_loader, criterion, optimizer, num_epochs=20):
        model.train()
        best_validation_loss = float('inf')
        no_improvement_count = 0
        for epoch in range(num_epochs):
            running_loss = 0.0
            for eeg_vectors, bert_vectors, labels in train_loader:
                optimizer.zero_grad()
                output1 = model(eeg_vectors)
                output2 = bert_vectors  # Assuming bert_vectors are treated as target embeddings

                print("labels shape", labels.shape)

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

    def evaluate(model, data_loader, criterion):
        model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for eeg_vectors, bert_vectors, labels in data_loader:
                output1 = model(eeg_vectors)
                output2 = bert_vectors
                loss = criterion(output1, output2, labels)
                total_loss += loss.item() * eeg_vectors.size(0)
                total_samples += eeg_vectors.size(0)

        return total_loss / total_samples


    train_contrastive(model, train_loader, criterion, optimizer)
