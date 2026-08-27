import torch.nn as nn
import torch
from data_prep import load_and_prepare
from models import GRUModel

def train_and_eval(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(X_test), y_test).item()
    return loss.item(), test_loss

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(lookback=10)

    print("--- Dropout YOK (0.0) ---")
    a = GRUModel(dropout=0.0)
    tr_a, te_a = train_and_eval(a, X_train, y_train, X_test, y_test)
    print(f"train_loss={tr_a:.6f} | test_loss={te_a:.6f}")

    print("\n--- Dropout VAR (0.3) ---")
    b = GRUModel(dropout=0.3)
    tr_b, te_b = train_and_eval(b, X_train, y_train, X_test, y_test)
    print(f"train_loss={tr_b:.6f} | test_loss={te_b:.6f}")
