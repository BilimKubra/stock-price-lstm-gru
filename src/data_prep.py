import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare(csv_path='data/amzn_prices.csv', lookback=20, train_ratio=0.8):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    close = df['Close'].astype(float).values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    close_scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(len(close_scaled) - lookback):
        X.append(close_scaled[i:i + lookback, 0])
        y.append(close_scaled[i + lookback, 0])
    X = np.array(X)
    y = np.array(y)

    # Kronolojik train/test ayrimi (karistirma YOK - zaman sirasi onemli)
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # PyTorch RNN sekli: (ornek, zaman_adimi, ozellik)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    return X_train_t, X_test_t, y_train_t, y_test_t, scaler

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = load_and_prepare()
    print('X_train shape:', X_train.shape, ' (ornek, zaman_adimi, ozellik)')
    print('X_test shape :', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape :', y_test.shape)
