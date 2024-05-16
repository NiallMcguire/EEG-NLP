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

# Example data (batch_size, input_size)
batch_size = 32
eeg_data = torch.randn(batch_size, 840)  # Example EEG data
bert_embedding = torch.randn(batch_size, 768)  # Example BERT embedding
labels = torch.randint(0, 3, (batch_size,))  # Random labels for named entity recognition

for epoch in range(num_epochs):
    # Concatenate EEG data and BERT embeddings
    combined_input = torch.cat((eeg_data, bert_embedding), dim=1)

    # Forward pass
    output = model(combined_input)

    # Calculate loss
    loss = criterion(output, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
