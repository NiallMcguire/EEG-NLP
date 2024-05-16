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
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, query_embeddings):
        # Compute pairwise cosine similarity matrix
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), query_embeddings.unsqueeze(0), dim=-1)

        # Construct positive and negative pairs
        mask = torch.eye(embeddings.size(0), dtype=torch.bool).to(embeddings.device)
        positives = sim_matrix.masked_select(mask).view(embeddings.size(0), -1)
        negatives = sim_matrix.masked_select(~mask).view(embeddings.size(0), -1)

        # Compute contrastive loss
        targets = torch.arange(embeddings.size(0)).to(embeddings.device)
        loss_pos = F.cross_entropy(positives, targets)
        loss_neg = F.cross_entropy(-negatives, targets)
        loss = loss_pos + loss_neg
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