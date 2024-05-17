import torch
import torch.nn as nn
import torch.optim as optim
from src import Networks




if __name__ == "main":
    # Adjusted input and output sizes
    input_size = 840 + 768  # EEG data vector size + BERT embedding size
    output_size = 3  # Number of classes for named entity recognition

    # Hyperparameters
    hidden_size = 50
    learning_rate = 0.001
    num_epochs = 20

    # Initialize the MLP model, loss function, and optimizer
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()  # Since it's a classification task, use CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)