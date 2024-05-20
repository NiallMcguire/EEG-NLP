import sys

sys.path.append('/users/gxb18167/EEG-NLP')

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


from src import Networks
from src import data
from src import utils
import numpy as np


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
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class EEGToBERTModel(nn.Module):
    def __init__(self, eeg_input_dim, bert_output_dim):
        super(EEGToBERTModel, self).__init__()
        self.fc1 = nn.Linear(eeg_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, bert_output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"
    #train_path = r"C:\Users\gxb18167\PycharmProjects\EEG-NLP\NER.pkl" #@TODO path change to above

    d = data.Data()
    util = utils.Utils()

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    vector_size = 768

    test_size = 0.2

    ner_bert = utils.NER_BERT()

    train_NE_embedded = ner_bert.get_embeddings(train_NE)

    train_NE_expanded = util.NER_expanded_NER_list(train_EEG_segments, train_NE_embedded, vector_size)

    train_NE_expanded = np.array(train_NE_expanded)

    #train_NE_padded_tensor = torch.tensor(train_NE_expanded, dtype=torch.float32)

    X, y = util.NER_padding_x_y(train_EEG_segments, train_Classes)
    X = np.array(X)
    X = util.NER_reshape_data(X)
    y_categorical = util.encode_labels(y)


    # Create pairs and labels
    positive_pairs = [(X[i], train_NE_expanded[i], 1) for i in range(len(X))]
    negative_pairs = []

    # Generate negative pairs
    for i in range(len(X)):
        for j in range(len(train_NE_expanded)):
            if i != j:
                negative_pairs.append((X[i], train_NE_expanded[j], 0))

    all_pairs = positive_pairs + negative_pairs

    #convert to tensors
    eeg_pairs = torch.tensor([pair[0] for pair in all_pairs], dtype=torch.float32)
    bert_pairs = torch.tensor([pair[1] for pair in all_pairs], dtype=torch.float32)
    labels = torch.tensor([pair[2] for pair in all_pairs], dtype=torch.float32)

    eeg_train, eeg_test, bert_train, bert_test, labels_train, labels_test = train_test_split(eeg_pairs, bert_pairs, labels, test_size=test_size, random_state=42)


    train_dataset = EEGToBERTContrastiveDataset(eeg_train, bert_train, labels_train)
    test_dataset = EEGToBERTContrastiveDataset(eeg_test, bert_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    print("EEG input shape: ", eeg_train.shape)
    print("BERT input shape: ", bert_train.shape)


    breakpoint()
    # Assuming the model is already defined as EEGToBERTModel

    eeg_input_dim = 128
    bert_output_dim = 768

    model = EEGToBERTModel(eeg_input_dim, bert_output_dim)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    def train_contrastive(model, train_loader, criterion, optimizer, num_epochs=20):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for eeg_vectors, bert_vectors, labels in train_loader:
                optimizer.zero_grad()
                output1 = model(eeg_vectors)
                output2 = bert_vectors  # Assuming bert_vectors are treated as target embeddings
                loss = criterion(output1, output2, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * eeg_vectors.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')


    train_contrastive(model, train_loader, criterion, optimizer)
