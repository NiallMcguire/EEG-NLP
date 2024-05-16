import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cosine_similarity = nn.functional.cosine_similarity(output1, output2, dim=-1)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cosine_similarity, 2) +
                                      label * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2))
        return loss_contrastive


# Hyperparameters
input_size = 100  # Example input feature size
hidden_size = 50
output_size = 10
learning_rate = 0.001
num_epochs = 20
margin = 1.0

# Initialize the MLP model, loss function, and optimizer
model = MLP(input_size, hidden_size, output_size)
criterion = ContrastiveLoss(margin=margin)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Example data (batch_size, input_size)
batch_size = 32
data1 = torch.randn(batch_size, input_size)
data2 = torch.randn(batch_size, input_size)
labels = torch.randint(0, 2, (batch_size,)).float()  # Random binary labels

for epoch in range(num_epochs):
    # Forward pass
    output1 = model(data1)
    output2 = model(data2)

    # Ensure the outputs are reshaped correctly for cosine similarity
    output1 = output1.view(batch_size, -1)
    output2 = output2.view(batch_size, -1)

    loss = criterion(output1, output2, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')