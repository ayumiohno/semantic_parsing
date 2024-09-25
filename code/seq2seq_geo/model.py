import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True
        )  # , dropout=0.2)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        h = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        c = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        _, (h, c) = self.lstm(x, (h, c))
        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True
        )  # , dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.embedding = nn.Embedding(num_classes, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, c, h):
        x = self.embedding(x)
        x = self.dropout(x)
        out, (h, c) = self.lstm(x, (h, c))
        out = self.dropout(out)
        out = self.fc(out.squeeze(1))
        pred = self.softmax(out)
        return pred, h, c
