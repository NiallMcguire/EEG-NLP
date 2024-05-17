import torch
import torch.nn as nn
import torch.optim as optim





if __name__ == "main":
    # Adjusted input and output sizes
    input_size = 840 + 768  # EEG data vector size + BERT embedding size
    output_size = 3  # Number of classes for named entity recognition

    # Hyperparameters
    hidden_size = 50
    learning_rate = 0.001
    num_epochs = 20