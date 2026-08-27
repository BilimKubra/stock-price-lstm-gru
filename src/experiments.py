import time
import csv
import torch
import torch.nn as nn
from data_prep import load_and_prepare
from models import LSTMModel, GRUModel

def train_and_eval(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01):
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
    elapsed = time.time() - start

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test).item()

    return loss.item(), test_loss, elapsed


if __name__ == '__main__':
    lookbacks = [10, 20, 30]
    model_types = {'LSTM': LSTMModel, 'GRU': GRUModel}
    results = []

    for lookback in lookbacks:
        X_train, X_test, y_train, y_test, scaler = load_and_prepare(lookback=lookback)
        for name, ModelClass in model_types.items():
            model = ModelClass()
            train_loss, test_loss, elapsed = train_and_eval(model, X_train, y_train, X_test, y_test)
            results.append({
                'lookback': lookback,
                'model': name,
                'hidden_size': 32,
                'train_loss': round(train_loss, 6),
                'test_loss': round(test_loss, 6),
                'sure_sn': round(elapsed, 2),
            })
            print(f"lookback={lookback:2d} | {name:4s} | train_loss={train_loss:.6f} | test_loss={test_loss:.6f} | sure={elapsed:.2f}s")

    with open('docs/experiment_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nSonuclar kaydedildi: docs/experiment_results.csv")
