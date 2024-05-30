import os

import torch
import torch.nn.functional as F
import numpy as np
import nltk
from torch.utils.data import Dataset

nltk.download('punkt')
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import json


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

    def max_dimension_size(self, arr):
        if arr.ndim == 1:
            return 1
        else:
            return arr.shape[0]

    def NER_expanded_NER_list(self, EEG_segments, NE, padding_shape=50):
        expanded_named_entity_list = []
        for i in range(len(EEG_segments)):
            named_entities = NE[i]
            for j in range(len(EEG_segments[i])):
                expanded_named_entity_list.append(named_entities)

        if type(NE[0]) != list:
            for i in range(len(expanded_named_entity_list)):
                if expanded_named_entity_list[i].shape == (padding_shape,):
                    expanded_named_entity_list[i] = expanded_named_entity_list[i].reshape(1, -1).tolist()
                else:
                    expanded_named_entity_list[i] = expanded_named_entity_list[i].tolist()

        max_seq_len = max([len(i) for i in expanded_named_entity_list])

        # padding function
        for i in range(len(expanded_named_entity_list)):
            list_element = expanded_named_entity_list[i]
            padding_count = max_seq_len - len(expanded_named_entity_list[i])
            for j in range(padding_count):
                list_element.append(np.zeros(padding_shape))

        return expanded_named_entity_list

    #util save json
    def save_json(self, data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f)

    #util load json
    def load_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def load_pre_training_gridsearch(self, models, config_paths):
        # list all files in directory
        model_save_paths = []
        model_names = []
        for file in os.listdir(config_paths):
            # check if file is a .json file and contains NER
            if file.endswith(".json") and "NER_Pre_Training" in file:
                parameter_dictionary = self.load_json(config_paths + file)
                if 'model_name' in parameter_dictionary:
                    for model in models:
                        if model in parameter_dictionary['model_name']:
                            #print(f"Loading model: {model}")
                            model_save_path = parameter_dictionary['model_save_path']
                            model_save_paths.append(model_save_path)
                            model_names.append(model)
        return model_save_paths, model_names



class NER_BERT:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_embeddings(self, Word_Labels_List):
        embedded_input = []
        for named_entity in Word_Labels_List:
            inputs = self.tokenizer(named_entity, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            embedded_input.append(last_hidden_states.mean(dim=1).squeeze().detach().numpy())

        return embedded_input

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