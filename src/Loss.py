import torch
import torch.nn as nn
import torch.nn.functional as F



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


class ContrastiveLossCosine(nn.Module):
    """
    Contrastive loss class using cosine similarity between embeddings

    Args:
        margin (float): Margin value for contrastive loss

    Returns:
        loss_contrastive (tensor): Contrastive loss value

    """
    def __init__(self, margin=1.0):
        super(ContrastiveLossCosine, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cosine_similarity = nn.functional.cosine_similarity(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(cosine_similarity, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2))
        return loss_contrastive




class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean(torch.clamp(distance_positive - distance_negative + self.margin, min=0.0))
        return loss







