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
    train_NE_embedded = ner_bert.get_embeddings(NE)