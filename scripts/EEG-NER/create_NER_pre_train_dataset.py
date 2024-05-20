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


if __name__ == "__main__":
    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"

    save_path = r"/users/gxb18167/EEG-NLP/"
    # train_path = r"C:\Users\gxb18167\PycharmProjects\EEG-NLP\NER.pkl" #@TODO path change to above

    d = data.Data()
    util = utils.Utils()

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    vector_size = 768

    test_size = 0.2

    ner_bert = utils.NER_BERT()

    train_NE_embedded = ner_bert.get_embeddings(train_NE)

    train_NE_expanded = util.NER_expanded_NER_list(train_EEG_segments, train_NE_embedded, vector_size)

    train_NE_expanded = np.array(train_NE_expanded)

    # train_NE_padded_tensor = torch.tensor(train_NE_expanded, dtype=torch.float32)

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

    del positive_pairs
    del negative_pairs
    del X
    del train_NE_expanded
    del train_NE_embedded
    del train_NE
    del train_EEG_segments

    # convert to tensors
    eeg_pairs = torch.tensor([pair[0] for pair in all_pairs], dtype=torch.float32)
    bert_pairs = torch.tensor([pair[1] for pair in all_pairs], dtype=torch.float32)
    labels = torch.tensor([pair[2] for pair in all_pairs], dtype=torch.float32)

    #save data
    torch.save(eeg_pairs, save_path + "Pre_NER_eeg_pairs.pt")
    torch.save(bert_pairs, save_path + "Pre_NER_bert_pairs.pt")
    torch.save(labels, save_path + "Pre_NER_labels.pt")

