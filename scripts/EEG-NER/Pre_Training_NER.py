import torch
import torch.nn as nn
import torch.optim as optim
from src import Networks
from src import data
from src import utils
import numpy as np



if __name__ == "main":
    #train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"
    train_path = r"C:\Users\gxb18167\PycharmProjects\EEG-NLP\NER.pkl" #@TODO path change to above

    d = data.Data()
    util = utils.Utils()

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

    vector_size = 768

    ner_bert = utils.NER_BERT()

    train_NE_embedded = ner_bert.get_embeddings(train_NE)

    train_NE_expanded = util.NER_expanded_NER_list(train_EEG_segments, train_NE_embedded, vector_size)

    train_NE_expanded = np.array(train_NE_expanded)

    train_NE_padded_tensor = torch.tensor(train_NE_expanded, dtype=torch.float32)

