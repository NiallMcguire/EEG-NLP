import torch
import torch.nn.functional as F


def encode_labels(y):
    label_encoder = {label: idx for idx, label in enumerate(set(y))}
    encoded_labels = [label_encoder[label] for label in y]

    # Step 2: Convert numerical labels to tensor
    encoded_labels_tensor = torch.tensor(encoded_labels)

    # Step 3: Convert numerical labels tensor to one-hot encoded tensor
    num_classes = len(label_encoder)
    y_onehot = F.one_hot(encoded_labels_tensor, num_classes=num_classes).float()

    return y_onehot