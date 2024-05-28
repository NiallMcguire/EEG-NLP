import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the BLSTM classifier model
class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.2):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out


class BLSTM_Text(nn.Module):
    def __init__(self, input_size, vector_size, hidden_size, num_classes, num_layers, dropout=0.2):
        super(BLSTM_Text, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size + vector_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, additional_input):
        # Concatenate additional_input with the original input along the feature dimension
        combined_input = torch.cat((x, additional_input), dim=2)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(combined_input, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out


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

class EEGToBERTModel_v1(nn.Module):
    def __init__(self, eeg_input_dim, bert_output_dim):
        super(EEGToBERTModel_v1, self).__init__()
        self.lstm = nn.LSTM(eeg_input_dim, bert_output_dim * 7, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class EEGToBERTModel_v2(nn.Module):
    def __init__(self, eeg_input_dim, bert_output_dim):
        super(EEGToBERTModel_v2, self).__init__()
        self.lstm = nn.LSTM(eeg_input_dim, 512, batch_first=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, bert_output_dim * 7)  # Adjusted output dimension

    def forward(self, x):
        # LSTM expects input of shape (batch_size, sequence_length, input_dim)
        x, _ = self.lstm(x)  # LSTM output and (hidden_state, cell_state)
        x = torch.relu(self.fc1(
            x[:, -1, :]))  # Take the output of the last time step and pass it through the fully connected layer
        x = self.fc2(x)
        # Reshape x to match the shape of output2
        x = x.view(x.size(0), 7, -1)
        return x

class EEGToBERTModel_v3(nn.Module):
    def __init__(self, eeg_input_dim, bert_output_dim):
        super(EEGToBERTModel_v3, self).__init__()
        self.lstm = nn.LSTM(eeg_input_dim, 512, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(1024, 256)  # Since LSTM is bidirectional, the output size is doubled
        self.fc2 = nn.Linear(256, bert_output_dim * 7)  # Adjusted output dimension

    def forward(self, x):
        # LSTM expects input of shape (batch_size, sequence_length, input_dim)
        x, _ = self.lstm(x)  # LSTM output and (hidden_state, cell_state)
        x = torch.relu(self.fc1(
            x[:, -1, :]))  # Take the output of the last time step and pass it through the fully connected layer
        x = self.fc2(x)
        # Reshape x to match the shape of output2
        x = x.view(x.size(0), 7, -1)
        return x

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        energy = self.attn(encoder_outputs.contiguous().view(-1, self.hidden_dim * 2))
        energy = energy.view(-1, seq_len)  # Reshape to (batch_size, seq_len)
        attention_weights = torch.softmax(energy, dim=1).unsqueeze(2)  # Add dimension for broadcasting
        context_vector = torch.sum(encoder_outputs * attention_weights, dim=1).unsqueeze(1)
        return context_vector

class EEGToBERTModel_v4(nn.Module):
    def __init__(self, eeg_input_dim, bert_output_dim, hidden_dim=512):
        super(EEGToBERTModel_v4, self).__init__()
        self.lstm = nn.LSTM(eeg_input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2 + hidden_dim, 256)  # Adjusted input dimension
        self.fc2 = nn.Linear(256, bert_output_dim)

    def forward(self, x):
        # LSTM expects input of shape (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)  # LSTM output and (hidden_state, cell_state)
        attn_output = self.attention(lstm_out)
        lstm_out = torch.cat((lstm_out[:, -1, :], attn_output.squeeze(1)), dim=1)  # Concatenate LSTM output and attention output
        lstm_out = torch.relu(self.fc1(lstm_out))  # Pass through fully connected layer
        output = self.fc2(lstm_out)  # Final output
        # Reshape output to match the shape of [batchsize, 7, bert_output_dim]
        output = output.view(output.size(0), 7, -1)
        return output