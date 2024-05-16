import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Define a custom dataset for contrastive learning
class CustomDataset(Dataset):
    def __init__(self, brain_embeddings, query_embeddings):
        self.brain_embeddings = brain_embeddings
        self.query_embeddings = query_embeddings

    def __len__(self):
        return len(self.brain_embeddings)

    def __getitem__(self, idx):
        return self.brain_embeddings[idx], self.query_embeddings



# Define a custom contrastive loss function
import torch.nn.functional as F
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, query_embeddings):
        # Ensure that embeddings and query_embeddings have compatible dimensions
        # embeddings: (batch_size, embedding_dim)
        # query_embeddings: (batch_size, num_tokens, embedding_dim)

        # Compute the cosine similarity
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), query_embeddings, dim=-1)

        # Compute the contrastive loss
        # Assume we have binary labels indicating whether pairs are positive or negative samples
        # For simplicity, we use a dummy label tensor (adjust according to your dataset)
        labels = torch.ones(embeddings.size(0)).to(embeddings.device)  # Example: all positive pairs

        # Positive pairs: loss = 1 - cos_sim
        pos_loss = (1 - sim_matrix[labels == 1]).pow(2).sum()

        # Negative pairs: loss = max(0, cos_sim - margin)
        neg_loss = (sim_matrix[labels == 0] - self.margin).clamp(min=0).pow(2).sum()

        loss = pos_loss + neg_loss
        return loss



class MLP(nn.Module):
    def __init__(self, input_dim, position_embedding_dim, hidden_layer_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.fc3 = nn.Linear(hidden_layer_dim, position_embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x