import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_prep import load_and_prepare
from models import LSTMModel, GRUModel

LOOKBACK = 10
EPOCHS = 100
LR = 0.01

def train_and_eval(model, X_train, y_train, X_test, y_test, scaler, name):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start

    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(X_test)

    mse_scaled = loss_fn(test_pred_scaled, y_test).item()

    test_pred_real = scaler.inverse_transform(test_pred_scaled.numpy())
    y_test_real = scaler.inverse_transform(y_test.numpy())
    rmse_dollar = np.sqrt(np.mean((test_pred_real - y_test_real) ** 2))

    print(f"[{name}] test_mse(normalize)={mse_scaled:.6f} | test_rmse($)={rmse_dollar:.2f} | sure={elapsed:.2f}s")
    return test_pred_real, y_test_real, mse_scaled, rmse_dollar, elapsed


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(lookback=LOOKBACK)

    torch.manual_seed(42)
    lstm = LSTMModel()
    lstm_pred, y_real, lstm_mse, lstm_rmse, lstm_time = train_and_eval(lstm, X_train, y_train, X_test, y_test, scaler, "LSTM")

    torch.manual_seed(42)
    gru = GRUModel()
    gru_pred, _, gru_mse, gru_rmse, gru_time = train_and_eval(gru, X_train, y_train, X_test, y_test, scaler, "GRU")

    print("\n=== Final Karsilastirma (lookback=10) ===")
    print(f"{'Model':6s} | {'Test MSE':>10s} | {'Test RMSE($)':>12s} | {'Sure(s)':>8s}")
    print(f"{'LSTM':6s} | {lstm_mse:10.6f} | {lstm_rmse:12.2f} | {lstm_time:8.2f}")
    print(f"{'GRU':6s} | {gru_mse:10.6f} | {gru_rmse:12.2f} | {gru_time:8.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(y_real, label='Gercek Fiyat', linewidth=2)
    plt.plot(lstm_pred, label='LSTM Tahmini', linestyle='--')
    plt.plot(gru_pred, label='GRU Tahmini', linestyle='--')
    plt.xlabel('Test Gun Indeksi')
    plt.ylabel('AMZN Kapanis Fiyati ($)')
    plt.title('Gercek vs Tahmin - LSTM vs GRU (lookback=10)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/lstm_vs_gru_predictions.png', dpi=150)
    print("\nGrafik kaydedildi: docs/lstm_vs_gru_predictions.png")
