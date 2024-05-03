import torch
import torch.nn as nn



class ContrastiveLossEuclid(nn.Module):
    """
    Contrastive loss class using euclidean distance between embeddings

    Args:
        margin (float): Margin value for contrastive loss

    Returns:
        loss_contrastive (tensor): Contrastive loss value

    """
    def __init__(self, margin=2.0):
        super(ContrastiveLossEuclid, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive