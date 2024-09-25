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
        self.dropout = nn.Dropout(0.5)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        h = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        c = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        out, (h, c) = self.lstm(x, (h, c))
        return out, h, c


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, encoder_outputs, h):
        weights = torch.bmm(
            encoder_outputs.unsqueeze(0), h.unsqueeze(0).transpose(1, 2)
        )
        s = torch.softmax(weights.squeeze(0), dim=0).unsqueeze(0)
        c = torch.bmm(encoder_outputs.unsqueeze(0).transpose(1, 2), s)
        h_attn = torch.tanh(
            self.attention(torch.cat((c.squeeze(0).transpose(0, 1), h), 1))
        )
        return h_attn


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
        self.dropout = nn.Dropout(0.5)
        # self.dropout = nn.Dropout(0.2)
        self.attention = Attention(hidden_size)

    def forward(self, x, c, h, encoder_outputs):
        x = self.embedding(x)
        x = self.dropout(x)
        out, (h, c) = self.lstm(x, (h, c))
        out = self.attention(encoder_outputs, out)
        out = self.dropout(out)
        out = self.fc(out.squeeze(1))
        pred = self.softmax(out)
        return pred, h, c
