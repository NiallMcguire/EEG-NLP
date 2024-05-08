import torch
import torch.nn.functional as F
import numpy as np


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


    def NER_padding_x_y(self, EEG_segments, Classes, named_entity_list):
        X = []
        y = []
        for i in range(len(EEG_segments)):
            named_entity = named_entity_list[i]
            label = Classes[i][0]
            #print(label)
            EEG_list = EEG_segments[i]
            for EEG in EEG_list:
                if EEG != []:
                    X.append(EEG)
                    y.append(label)
        max_seq_length = max([len(x) for x in X])
        #paddding
        for i in range(len(X)):
            padding_count = max_seq_length - len(X[i])
            #print(padding_count)
            for j in range(padding_count):
                X[i].append(np.zeros((105,8)))

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