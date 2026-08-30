import numpy as np
from data_prep import load_and_prepare

LOOKBACK = 10

def run_naive_baseline(lookback=LOOKBACK, csv_path='data/amzn_prices.csv'):
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(csv_path=csv_path, lookback=lookback)
    # Naive tahmin: yarinki fiyat = bugunku fiyat (pencerenin son gozlemi)
    naive_pred_scaled = X_test[:, -1, :].numpy()
    y_test_scaled = y_test.numpy()

    naive_pred_real = scaler.inverse_transform(naive_pred_scaled)
    y_test_real = scaler.inverse_transform(y_test_scaled)

    rmse_dollar = np.sqrt(np.mean((naive_pred_real - y_test_real) ** 2))
    return rmse_dollar

if __name__ == '__main__':
    rmse = run_naive_baseline()
    print(f"[Naive Baseline] test_rmse($)={rmse:.2f}")
