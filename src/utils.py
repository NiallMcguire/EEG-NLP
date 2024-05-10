import torch
import torch.nn.functional as F
import numpy as np
import nltk
nltk.download('punkt')
from gensim.models import Word2Vec


class Utils:

    def __init__(self):
        pass

    def encode_labels(self, y):
        label_encoder = {label: idx for idx, label in enumerate(set(y))}
        encoded_labels = [label_encoder[label] for label in y]

        # Step 2: Convert numerical labels to tensor
        encoded_labels_tensor = torch.tensor(encoded_labels)

        # Step 3: Convert numerical labels tensor to one-hot encoded tensor
        num_classes = len(label_encoder)
        y_onehot = F.one_hot(encoded_labels_tensor, num_classes=num_classes).float()

        return y_onehot


    def NER_padding_x_y(self, segments, Classes, padding_shape=(105,8)):
        X = []
        y = []
        for i in range(len(segments)):
            label = Classes[i][0]
            EEG_list = segments[i]
            for EEG in EEG_list:
                if EEG != []:
                    X.append(EEG)
                    y.append(label)
        max_seq_length = max([len(x) for x in X])

        #paddding
        for i in range(len(X)):
            padding_count = max_seq_length - len(X[i])
            for j in range(padding_count):
                X[i].append(np.zeros(padding_shape))
        return X, y

    def NER_reshape_data(self, X):
        # reshape the data to 840
        new_list = []
        for i in range(len(X)):
            array_list = X[i]
            arrays_list_reshaped = [arr.reshape(-1) for arr in array_list]
            new_list.append(arrays_list_reshaped)
        new_list = np.array(new_list)
        return new_list

    def NER_Word2Vec(self, Word_Labels_List, vector_size=50, window=5, min_count=1, workers=4):

        model = Word2Vec(sentences=Word_Labels_List, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}

        embedded_input = []
        for named_entity in Word_Labels_List:
            sequence = []
            for word in named_entity:
                sequence.append(word_embeddings[word])

            embedded_input.append(sequence)

        return word_embeddings, embedded_input

    def NER_expanded_NER_list(self, EEG_segments, NE, padding_shape=(50,)):
        expanded_named_entity_list = []
        for i in range(len(EEG_segments)):
            named_entities = NE[i]
            for j in range(len(EEG_segments[i])):
                expanded_named_entity_list.append(named_entities)

        max_seq_len = max([len(i) for i in expanded_named_entity_list])
        # padding function
        for i in range(len(expanded_named_entity_list)):
            padding_count = max_seq_len - len(expanded_named_entity_list[i])
            for j in range(padding_count):
                expanded_named_entity_list[i].append(np.zeros(padding_shape))
        return expanded_named_entity_list