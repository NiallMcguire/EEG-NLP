import sys

sys.path.append('/users/gxb18167/EEG-NLP')

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from src import data
from src import utils
import numpy as np
from random import sample


from torch.utils.data import Dataset, DataLoader

class EEGToBERTContrastiveDataset(Dataset):
    def __init__(self, eeg_data, bert_data, labels):
        self.eeg_data = eeg_data
        self.bert_data = bert_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg_vector = self.eeg_data[idx]
        bert_vector = self.bert_data[idx]
        label = self.labels[idx]
        return eeg_vector, bert_vector, label



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        # Reshape label tensor to match the shape of euclidean_distance tensor
        label = label.unsqueeze(1).expand_as(euclidean_distance)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class EEGToBERTModel(nn.Module):
    def __init__(self, eeg_input_dim, bert_output_dim):
        super(EEGToBERTModel, self).__init__()
        self.lstm = nn.LSTM(eeg_input_dim, 512, batch_first=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, bert_output_dim * 7)  # Adjusted output dimension

    def forward(self, x):
        # LSTM expects input of shape (batch_size, sequence_length, input_dim)
        x, _ = self.lstm(x)  # LSTM output and (hidden_state, cell_state)
        x = torch.relu(self.fc1(
            x[:, -1, :]))  # Take the output of the last time step and pass it through the fully connected layer
        x = self.fc2(x)
        # Reshape x to match the shape of output2
        x = x.view(x.size(0), 7, -1)
        return x

if __name__ == "__main__":
    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"
    #save_path = r"/users/gxb18167/EEG-NLP/"

    d = data.Data()
    util = utils.Utils()

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    vector_size = 768
    test_size = 0.2

    ner_bert = utils.NER_BERT()

    train_NE_embedded = ner_bert.get_embeddings(train_NE)

    train_NE_expanded = util.NER_expanded_NER_list(train_EEG_segments, train_NE_embedded, vector_size)
    train_NE_expanded = np.array(train_NE_expanded)

    X, y = util.NER_padding_x_y(train_EEG_segments, train_Classes)
    X = np.array(X)
    X = util.NER_reshape_data(X)
    y_categorical = util.encode_labels(y)

    positive_pairs = [(X[i], train_NE_expanded[i], 1) for i in range(len(X))]

    num_negative_pairs_per_positive = 1
    negative_pairs = []

    for i in range(len(X)):
        negative_indices = sample([j for j in range(len(train_NE_expanded)) if j != i], num_negative_pairs_per_positive)
        for neg_index in negative_indices:
            negative_pairs.append((X[i], train_NE_expanded[neg_index], 0))

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

    #validation
    eeg_train, eeg_val, bert_train, bert_val, labels_train, labels_val = train_test_split(eeg_train, bert_train, labels_train, test_size=0.2, random_state=42)


    train_dataset = EEGToBERTContrastiveDataset(eeg_train, bert_train, labels_train)
    validation_dataset = EEGToBERTContrastiveDataset(eeg_val, bert_val, labels_val)

    test_dataset = EEGToBERTContrastiveDataset(eeg_test, bert_test, labels_test)

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
    model = EEGToBERTModel(eeg_input_dim, bert_output_dim)
    criterion = ContrastiveLoss(margin=1.0)
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
