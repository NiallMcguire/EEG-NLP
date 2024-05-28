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

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, encoder_outputs):
        energy = torch.tanh(self.attn(encoder_outputs))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention, dim=1)

class EEGToBERTModel_v3(nn.Module):
    def __init__(self, eeg_input_dim, bert_output_dim):
        super(EEGToBERTModel_v3, self).__init__()
        self.lstm = nn.LSTM(eeg_input_dim, 256, batch_first=True, bidirectional=True)
        self.attention = Attention(256)
        self.fc = nn.Linear(256, bert_output_dim * 7)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        output = self.fc(context)
        output = output.view(output.size(0), 7, -1)
        return output
