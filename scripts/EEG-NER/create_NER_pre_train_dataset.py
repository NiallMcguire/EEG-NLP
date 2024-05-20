import sys
sys.path.append('/users/gxb18167/EEG-NLP')

import torch
import torch.optim as optim
import numpy as np
from src import Networks
from src import data
from src import utils

from torch.utils.data import DataLoader

if __name__ == "__main__":
    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"
    save_path = r"/users/gxb18167/EEG-NLP/"

    # Instantiate necessary objects
    d = data.Data()
    util = utils.Utils()

    # Read data from file
    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    # Calculate vector size
    vector_size = 768

    # Initialize NER_BERT object
    ner_bert = utils.NER_BERT()

    # Generate BERT embeddings
    train_NE_embedded = ner_bert.get_embeddings(train_NE)

    # Generate expanded NER list
    train_NE_expanded = util.NER_expanded_NER_list(train_EEG_segments, train_NE_embedded, vector_size)

    # Convert data to numpy arrays
    X, y = util.NER_padding_x_y(train_EEG_segments, train_Classes)
    X = util.NER_reshape_data(X)
    y_categorical = util.encode_labels(y)

    # Create positive pairs
    positive_pairs = [(X[i], train_NE_expanded[i], 1) for i in range(len(X))]

    # Create negative pairs
    negative_pairs = []
    for i in range(len(X)):
        for j in range(len(train_NE_expanded)):
            if i != j:
                negative_pairs.append((X[i], train_NE_expanded[j], 0))

    # Combine positive and negative pairs
    all_pairs = positive_pairs + negative_pairs

    # Convert pairs to tensors and save data to disk
    for idx, pair_type in enumerate(["eeg_pairs", "bert_pairs", "labels"]):
        data_to_save = torch.tensor([pair[idx] for pair in all_pairs], dtype=torch.float32)
        torch.save(data_to_save, save_path + f"Pre_NER_{pair_type}.pt")
