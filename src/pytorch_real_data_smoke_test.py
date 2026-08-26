import pandas as pd
import torch
import torch.nn as nn

# Gercek veriyi oku
df = pd.read_csv('data/amzn_prices.csv', index_col=0, parse_dates=True)
close = df['Close'].astype(float).values

# Bugunku fiyat -> yarinki fiyat ciftleri (henuz sliding window yok, o Asama 4'te)
x = torch.tensor(close[:-1], dtype=torch.float32).unsqueeze(1)
y = torch.tensor(close[1:], dtype=torch.float32).unsqueeze(1)

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

model = TinyModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 40 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.2f}")

print(f"\nSon gercek fiyat: {close[-1]:.2f}")
with torch.no_grad():
    tahmin = model(torch.tensor([[close[-1]]], dtype=torch.float32))
print(f"Modelin bir sonraki gun tahmini: {tahmin.item():.2f}")
