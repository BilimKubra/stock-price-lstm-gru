"""Rastgele baslangic tohumu deneyi: sonuc sans mi, tekrarlanabilir mi?

NEDEN VAR
---------
Sinir agi agirliklari rastgele baslatilir. Tek bir kosuda GRU'nun LSTM'i
gecmesi, o rastgele baslangicin sansi olabilir. README'de "tek kosu,
istatistiksel tekrar yapilmadi" diye yazan sinirlama tam olarak budur.

Bu script ayni deneyi BES farkli tohumla tekrarlar ve sunlari raporlar:
  * her tohum icin RMSE (dolar) ve sure
  * model basina ortalama +/- standart sapma
  * GRU kac tohumda LSTM'i gecti (5'te kac)
  * naive baseline ile karsilastirma (baseline deterministik, tek deger)

Kullanim:
    python3 src/tohum_deneyi.py

Cikti: docs/tohum_sonuclari.csv
"""
import csv
import time
import numpy as np
import torch
import torch.nn as nn

from data_prep import load_and_prepare
from models import LSTMModel, GRUModel
from baseline import run_naive_baseline

LOOKBACK = 10
EPOCHS = 100
LR = 0.01
TOHUMLAR = [42, 1, 7, 123, 2024]


def egit_ve_olc(ModelSinifi, X_train, y_train, X_test, y_test, scaler, tohum):
    """Tek bir tohumla egitir, dolar cinsinden RMSE ve sure dondurur."""
    torch.manual_seed(tohum)
    np.random.seed(tohum)

    model = ModelSinifi()
    kayip_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    baslangic = time.time()
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        tahmin = model(X_train)
        kayip = kayip_fn(tahmin, y_train)
        kayip.backward()
        optimizer.step()
    sure = time.time() - baslangic

    model.eval()
    with torch.no_grad():
        test_tahmin = model(X_test)

    tahmin_dolar = scaler.inverse_transform(test_tahmin.numpy())
    gercek_dolar = scaler.inverse_transform(y_test.numpy())
    rmse = float(np.sqrt(np.mean((tahmin_dolar - gercek_dolar) ** 2)))
    return rmse, sure


def main():
    X_train, X_test, y_train, y_test, scaler = load_and_prepare(lookback=LOOKBACK)
    baseline_rmse = run_naive_baseline(lookback=LOOKBACK)

    modeller = {"LSTM": LSTMModel, "GRU": GRUModel}
    satirlar = []
    toplam = {ad: [] for ad in modeller}

    print(f"Tohum deneyi | lookback={LOOKBACK} | epoch={EPOCHS} | {len(TOHUMLAR)} tohum\n")
    print(f"{'Tohum':>6s} | {'LSTM RMSE($)':>13s} | {'GRU RMSE($)':>12s} | {'Kazanan':>8s}")
    print("-" * 52)

    gru_galibiyet = 0
    for tohum in TOHUMLAR:
        satir = {"tohum": tohum}
        for ad, Sinif in modeller.items():
            rmse, sure = egit_ve_olc(Sinif, X_train, y_train, X_test, y_test, scaler, tohum)
            toplam[ad].append(rmse)
            satir[f"{ad.lower()}_rmse"] = round(rmse, 2)
            satir[f"{ad.lower()}_sure"] = round(sure, 2)
        kazanan = "GRU" if satir["gru_rmse"] < satir["lstm_rmse"] else "LSTM"
        if kazanan == "GRU":
            gru_galibiyet += 1
        satir["kazanan"] = kazanan
        satirlar.append(satir)
        print(f"{tohum:>6d} | {satir['lstm_rmse']:>13.2f} | {satir['gru_rmse']:>12.2f} | {kazanan:>8s}")

    print("\n=== OZET ===")
    print(f"{'Model':>8s} | {'Ortalama':>9s} | {'Std sapma':>10s} | {'En iyi':>7s} | {'En kotu':>8s}")
    print("-" * 54)
    for ad in modeller:
        d = np.array(toplam[ad])
        print(f"{ad:>8s} | {d.mean():9.2f} | {d.std():10.2f} | {d.min():7.2f} | {d.max():8.2f}")
    print(f"{'Baseline':>8s} | {baseline_rmse:9.2f} | {'—':>10s} | {'—':>7s} | {'—':>8s}")

    print(f"\nGRU, {len(TOHUMLAR)} tohumun {gru_galibiyet}'inde LSTM'i gecti.")
    gru_ort = np.mean(toplam["GRU"])
    if baseline_rmse < gru_ort:
        fark = (gru_ort - baseline_rmse) / baseline_rmse * 100
        print(f"Naive baseline ({baseline_rmse:.2f}$) GRU ortalamasindan "
              f"%{fark:.0f} daha iyi - tek kosuluk bir tesaduf degil.")
    else:
        print("Bu kosuda GRU ortalamasi baseline'i gecti - onceki sonucla celisiyor, incelenmeli.")

    with open("docs/tohum_sonuclari.csv", "w", newline="") as f:
        yazici = csv.DictWriter(f, fieldnames=satirlar[0].keys())
        yazici.writeheader()
        yazici.writerows(satirlar)
    print("\nKaydedildi: docs/tohum_sonuclari.csv")


if __name__ == "__main__":
    main()
