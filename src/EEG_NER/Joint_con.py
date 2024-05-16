import torch
import torch.nn as nn
import torch.optim as optim


class JointEmbedding(nn.Module):
    def __init__(self, eeg_input_size, bert_input_size, hidden_size, output_size):
        super(JointEmbedding, self).__init__()
        self.eeg_blstm = nn.LSTM(input_size=eeg_input_size, hidden_size=hidden_size,
                                 num_layers=1, batch_first=True, bidirectional=True)
        self.bert_fc = nn.Linear(bert_input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 due to bidirectionality

    def forward(self, eeg_input, bert_input):
        # Encode EEG data with BLSTM
        eeg_out, _ = self.eeg_blstm(eeg_input)

        # Process BERT input
        bert_out = self.relu(self.bert_fc(bert_input))

        # Combine representations
        combined = eeg_out[:, -1, :] + bert_out  # Summing up the last BLSTM output and BERT output
        # Final classification layer
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
seq_length = 100  # Sequence length for BLSTM
eeg_data = torch.randn(batch_size, seq_length, eeg_input_size)  # Example EEG data
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
