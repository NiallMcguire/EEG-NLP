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


def NER_EEGtoEEG_create_paris(EEG_X, named_entity_class, max_positive_pairs=20000, max_negative_pairs=20000):
    """
    Create pairs of EEG samples and their contrastive labels with limits on the number of positive and negative pairs.

    Args:
    - EEG_X (array-like): Array of EEG samples.
    - named_entity_class (array-like): Array of named entity labels for EEG samples.
    - max_positive_pairs (int): Maximum number of positive pairs to include.
    - max_negative_pairs (int): Maximum number of negative pairs to include.

    Returns:
    - pairs (list of tuples): List of tuples where each tuple contains a pair of EEG samples.
    - labels (array-like): Array of contrastive labels indicating similarity (1) or dissimilarity (0).
    """
    pairs = []
    labels = []

    num_samples = len(EEG_X)
    positive_pairs = 0
    negative_pairs = 0

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            EEG_sample_i = EEG_X[i]
            EEG_sample_j = EEG_X[j]
            label_i = named_entity_class[i]
            label_j = named_entity_class[j]

            if label_i == label_j:
                # Positive pair
                if positive_pairs < max_positive_pairs:
                    pairs.append((EEG_sample_i, EEG_sample_j))
                    labels.append(1)
                    positive_pairs += 1
            else:
                # Negative pair
                if negative_pairs < max_negative_pairs:
                    pairs.append((EEG_sample_i, EEG_sample_j))
                    labels.append(0)
                    negative_pairs += 1

    return np.array(pairs), np.array(labels)


def NER_EEGtoBERT_create_pairs(EEG_X, NE_Expanded, named_entity_class, max_positive_pairs=20000, max_negative_pairs=20000):
    """
    Create pairs of EEG samples and BERT text embeddings with contrastive labels.

    Args:
    - EEG_X (array-like): Array of EEG samples.
    - NE_Expanded (array-like): Array of BERT embeddings corresponding to named entities.
    - named_entity_class (array-like): Array of named entity labels for EEG samples.
    - max_positive_pairs (int): Maximum number of positive pairs to include.
    - max_negative_pairs (int): Maximum number of negative pairs to include.

    Returns:
    - pairs (list of tuples): List of tuples where each tuple contains an EEG sample and a BERT embedding.
    - labels (array-like): Array of contrastive labels indicating similarity (1) or dissimilarity (0).
    """
    pairs = []
    labels = []

    num_samples = len(EEG_X)
    positive_pairs = 0
    negative_pairs = 0

    for i in range(num_samples):
        EEG_sample_i = EEG_X[i]
        label_i = named_entity_class[i]

        for j in range(num_samples):
            BERT_embedding_j = NE_Expanded[j]
            label_j = named_entity_class[j]

            if label_i == label_j:
                # Positive pair
                if positive_pairs < max_positive_pairs:
                    pairs.append((EEG_sample_i, BERT_embedding_j))
                    labels.append(1)
                    positive_pairs += 1
            else:
                # Negative pair
                if negative_pairs < max_negative_pairs:
                    pairs.append((EEG_sample_i, BERT_embedding_j))
                    labels.append(0)
                    negative_pairs += 1

    return np.array(pairs), np.array(labels)

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


if __name__ == "__main__":
    #data_path = r"C:\Users\gxb18167\PycharmProjects\EEG-NLP\NER.pkl"
    data_path = r"/users/gxb18167/EEG-NLP/NER.pkl"

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    d = data.Data()
    util = utils.Utils()
    Loss = Loss
    Networks = Networks

    NE, EEG_segments, Classes = d.NER_read_custom_files(data_path)

    parameters = {}

    max_positive_pairs = 20000
    max_negative_pairs = 20000
    contrastive_learning_setting = 'EEGtoEEG' #EEGtoBERT
    vector_size = 768
    parameters['vector_size'] = vector_size
    ner_bert = utils.NER_BERT()

    EEG_X, named_entity_class = util.NER_padding_x_y(EEG_segments, Classes)
    EEG_X = np.array(EEG_X)
    EEG_X = util.NER_reshape_data(EEG_X)
    named_entity_class_categorical = util.encode_labels(named_entity_class)

    if contrastive_learning_setting == "EEGtoBERT":
        NE_embedded = ner_bert.get_embeddings(NE)
        NE_expanded = util.NER_expanded_NER_list(EEG_segments, NE_embedded, vector_size)
        NE_expanded = np.array(NE_expanded)
        pairs, labels = NER_EEGtoBERT_create_pairs(EEG_X, NE_expanded, named_entity_class, max_positive_pairs, max_negative_pairs)
        print("Created EEG to BERT pairs of shape: ", pairs.shape)

    elif contrastive_learning_setting == "EEGtoEEG":
        pairs, labels = NER_EEGtoEEG_create_paris(EEG_X, named_entity_class, max_positive_pairs, max_negative_pairs)
        print("Created EEG to EEG pairs of shape: ", pairs.shape)


    # Convert to tensors
    pair_one = pairs[:, 0]
    pair_two = pairs[:, 1]
    labels = torch.tensor(labels, dtype=torch.float32)
    pair_one = torch.tensor(pair_one, dtype=torch.float32)
    pair_two = torch.tensor(pair_two, dtype=torch.float32)

    # Split data into training and testing sets using tensors
    pair_one_train, pair_one_test, pair_two_train, pair_two_test, labels_train, labels_test = train_test_split(pair_one, pair_two, labels, test_size=0.2, random_state=42)

    # Split training data into training and validation sets using tensors
    pair_one_train, pair_one_val, pair_two_train, pair_two_val, labels_train, labels_val = train_test_split(pair_one_train, pair_two_train, labels_train, test_size=0.1, random_state=42)

    # Create datasets
    train_dataset = utils.EEGContrastiveDataset(pair_one_train, pair_two_train, labels_train)
    val_dataset = utils.EEGContrastiveDataset(pair_one_val, pair_two_val, labels_val)
    test_dataset = utils.EEGContrastiveDataset(pair_one_test, pair_two_test, labels_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Print shapes
    print("Training data shapes: ", pair_one_train.shape, pair_two_train.shape, labels_train.shape)

    # Initialize model
    model = Networks.SiameseNetwork_v3().to(device)
    criterion = ContrastiveLoss(margin=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 100
    # early stopping
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            pair_one_batch, pair_two_batch, label_batch = batch
            pair_one_batch, pair_two_batch, label_batch = pair_one_batch.to(device), pair_two_batch.to(device), label_batch.to(device)

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
                pair_one_batch, pair_two_batch, label_batch = pair_one_batch.to(device), pair_two_batch.to(device), label_batch.to(device)

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















