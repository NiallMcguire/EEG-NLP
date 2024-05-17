import sys

sys.path.append('/users/gxb18167/EEG-NLP')

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


from src import Networks
from src import data
from src import utils
import numpy as np



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

    train_NE_padded_tensor = torch.tensor(train_NE_expanded, dtype=torch.float32)

    X, y = util.NER_padding_x_y(train_EEG_segments, train_Classes)
    X = np.array(X)
    X = util.NER_reshape_data(X)
    y_categorical = util.encode_labels(y)


    # Create pairs and labels
    positive_pairs = [(X[i], train_NE_padded_tensor[i], 1) for i in range(len(X))]
    negative_pairs = []

    # Generate negative pairs
    for i in range(len(X)):
        for j in range(len(train_NE_padded_tensor)):
            if i != j:
                negative_pairs.append((X[i], train_NE_padded_tensor[j], 0))

    breakpoint(print("Done"))



    train_NE_padded_tensor, test_NE_padded_tensor, _, _ = train_test_split(
        train_NE_padded_tensor, y_categorical, test_size=test_size, random_state=42)



    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=42)
