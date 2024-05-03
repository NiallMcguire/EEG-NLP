import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.EEG_NER_Pre_Training.Pre_Training import CustomDataset, ContrastiveLoss, MLP





if __name__ == "__main__":


    # Define the pre-training parameters
    num_brain_embeddings = 1000  # Number of brain embeddings
    num_query_embeddings = 500  # Number of query embeddings
    embedding_dim = 128  # Embedding dimensionality
    num_tokens = 100  # Number of tokens in text query
    num_time_frames = 50  # Number of time frames in brain representations
    position_embedding_dim = 64  # Dimensionality of position embeddings
    hidden_layer_dim = 256  # Dimensionality of hidden layers in MLP
    learning_rate = 0.01  # Learning rate for optimization
    num_iterations = 1000  # Number of optimization iterations
    batch_size = 32  # Batch size for contrastive learning

    # Initialize brain embeddings (bi) and query embeddings (vQ) using PyTorch tensors
    bi = torch.rand(num_time_frames, embedding_dim, requires_grad=True)
    vQ = torch.rand(num_query_embeddings, num_tokens, embedding_dim)

    # Create custom datasets and data loaders for contrastive learning
    custom_dataset = CustomDataset(bi, vQ)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the MLP network
    model = MLP(embedding_dim, position_embedding_dim, hidden_layer_dim)

    # Define the optimizer and contrastive loss function
    optimizer = optim.SGD([bi], lr=learning_rate)  # Only optimize brain embeddings (bi)
    criterion = ContrastiveLoss()