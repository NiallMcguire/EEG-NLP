import torch
import torch.nn as nn
import torch.optim as optim


class JointEmbedding(nn.Module):
    def __init__(self, eeg_input_size, bert_input_size, hidden_size, output_size):
        super(JointEmbedding, self).__init__()
        self.eeg_fc = nn.Linear(eeg_input_size, hidden_size)
        self.bert_fc = nn.Linear(bert_input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, eeg_input, bert_input):
        eeg_out = self.relu(self.eeg_fc(eeg_input))
        bert_out = self.relu(self.bert_fc(bert_input))
        combined = eeg_out + bert_out  # Sum or concatenate, depending on your data and task
        out = self.fc(combined)
        return out

# Adjusted input and output sizes
eeg_input_size = 840
bert_input_size = 768
output_size = 3  # Number of classes for named entity recognition

# Hyperparameters
hidden_size = 50
learning_rate = 0.001
num_epochs = 20

# Initialize the joint embedding model, loss function, and optimizer
model = JointEmbedding(eeg_input_size, bert_input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # Since it's a classification task, use CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Example data (batch_size, input_size)
batch_size = 32
eeg_data = torch.randn(batch_size, eeg_input_size)  # Example EEG data
bert_embedding = torch.randn(batch_size, bert_input_size)  # Example BERT embedding
labels = torch.randint(0, output_size, (batch_size,))  # Random labels for named entity recognition

for epoch in range(num_epochs):
    # Forward pass
    output = model(eeg_data, bert_embedding)

    # Calculate loss
    loss = criterion(output, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
