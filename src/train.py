import time
import torch
import torch.nn as nn
from data_prep import load_and_prepare
from models import LSTMModel, GRUModel

def train_model(model, X_train, y_train, epochs=100, lr=0.01, name="Model"):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[{name}] epoch {epoch:3d} | train loss {loss.item():.6f}")

    elapsed = time.time() - start
    print(f"[{name}] egitim suresi: {elapsed:.2f} saniye\n")
    return model, elapsed


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = load_and_prepare()

    print("=== LSTM egitimi ===")
    lstm = LSTMModel()
    lstm, lstm_time = train_model(lstm, X_train, y_train, epochs=100, name="LSTM")

    print("=== GRU egitimi ===")
    gru = GRUModel()
    gru, gru_time = train_model(gru, X_train, y_train, epochs=100, name="GRU")

    print(f"Ozet -> LSTM: {lstm_time:.2f}s | GRU: {gru_time:.2f}s")
