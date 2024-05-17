import torch
import torch.nn as nn
import torch.optim as optim
from src import Networks
from src import data




if __name__ == "main":
    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"

    d = data.Data()
    util = utils.Utils()

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)

