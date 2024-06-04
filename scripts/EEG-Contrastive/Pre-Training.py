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


if __name__ == "__main__":
    #data_path = r"C:\Users\gxb18167\PycharmProjects\EEG-NLP\NER.pkl"
    data_path = r"/users/gxb18167/EEG-NLP/NER.pkl"

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

    elif contrastive_learning_setting == "EEGtoEEG":
        pairs, labels = NER_EEGtoEEG_create_paris(EEG_X, named_entity_class, max_positive_pairs, max_negative_pairs)

        print(len(pairs))


    '''
    if parameters['Contrastive_label_setting'] == 'EEGtoBERT':
        # Create positive and negative pairs
        positive_pairs = [(EEG_X[i], NE_expanded[i], 1) for i in range(len(EEG_X))]

        negative_pairs = []
        for i in range(len(EEG_X)):
            negative_indices = sample([j for j in range(len(NE_expanded)) if j != i],
                                      num_negative_pairs_per_positive)
            for neg_index in negative_indices:
                negative_pairs.append((EEG_X[i], NE_expanded[neg_index], 0))
    '''


