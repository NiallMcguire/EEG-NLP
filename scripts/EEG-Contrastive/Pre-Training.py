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
    data_path = r"C:\Users\gxb18167\PycharmProjects\EEG-NLP\NER.pkl"


    d = data.Data()
    util = utils.Utils()
    Loss = Loss
    Networks = Networks

    NE, EEG_segments, Classes = d.NER_read_custom_files(data_path)

    parameters = {}

    vector_size = 768
    parameters['vector_size'] = vector_size
    ner_bert = utils.NER_BERT()
    NE_embedded = ner_bert.get_embeddings(NE)

    NE_expanded = util.NER_expanded_NER_list(EEG_segments, NE_embedded, vector_size)
    NE_expanded = np.array(NE_expanded)

    EEG_X, named_entity_class = util.NER_padding_x_y(EEG_segments, Classes)
    EEG_X = np.array(EEG_X)
    EEG_X = util.NER_reshape_data(EEG_X)
    named_entity_class_categorical = util.encode_labels(named_entity_class)

    print(EEG_X.shape)

