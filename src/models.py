import torch
import torch.nn as nn

INPUT_SIZE = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 2
OUTPUT_SIZE = 1

class LSTMModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])


if __name__ == '__main__':
    from data_prep import load_and_prepare
    X_train, X_test, y_train, y_test, scaler = load_and_prepare()

    lstm = LSTMModel()
    gru = GRUModel()

    lstm_out = lstm(X_train[:4])
    gru_out = gru(X_train[:4])

    print('LSTM cikti sekli:', lstm_out.shape, '(beklenen: [4, 1])')
    print('GRU cikti sekli :', gru_out.shape, '(beklenen: [4, 1])')
