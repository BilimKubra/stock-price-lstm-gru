# Stock Price Prediction: LSTM vs GRU

PyTorch ile zaman serisi regresyonu — AMZN hisse senedinin günlük kapanış
fiyatını, geçmiş bir pencereden yola çıkarak tahmin eden iki farklı sıralı
veri mimarisinin (LSTM ve GRU) uygulanması ve karşılaştırılması.

## Amaç

Bu proje, temel bir makine öğrenmesi kavramından (regresyon) başlayıp
PyTorch'ta LSTM ve GRU modellerini sıfırdan kurmayı, sistematik bir
deney süreciyle (Ar-Ge) hiperparametre kararları vermeyi ve iki mimariyi
adil koşullarda karşılaştırmayı amaçlıyor.

## Veri Seti

- Kaynak: [yfinance](https://github.com/ranaroussi/yfinance) (canlı API, ücretsiz)
- Hisse: AMZN (Amazon)
- Aralık: Son 2 yıl, günlük kapanış (Close) fiyatı
- Kaydedilmiş dosya: `data/amzn_prices.csv` (reproducibility için canlı çekimden sonra kaydedildi)

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Kullanım

```bash
python3 src/data_prep.py     # veri hazırlığını test et (normalize + sliding window)
python3 src/models.py        # model mimarilerini test et
python3 src/train.py         # tek seferlik egitim (baseline)
python3 src/experiments.py   # lookback deney matrisi (10/20/30 x LSTM/GRU)
python3 src/evaluate.py      # final degerlendirme + grafik
python3 src/baseline.py      # naive baseline (yarin=bugun) karsilastirmasi
pytest tests/                # birim testlerini calistir
```

Adim adim anlatili versiyon icin: [`notebooks/stock_prediction_lstm_gru.ipynb`](notebooks/stock_prediction_lstm_gru.ipynb)

## Mimari

| Parametre       | Değer |
|-----------------|-------|
| Lookback penceresi | 20 (baseline) / **10 (final)** |
| Hidden size     | 32 |
| Katman sayısı   | 2 |
| Loss / Optimizer | MSELoss / Adam |
| Epoch           | 100 |

Kararların gerekçesi için: [`docs/approach.md`](docs/approach.md)

## Sonuçlar

Final karşılaştırma (lookback=10, test seti üzerinde):

| Model | Test RMSE ($) | Eğitim Süresi (s) |
|-------|---------------|--------------------|
| Naive Baseline (yarın=bugün) | **5.97** | — |
| LSTM  | 11.94         | 0.49               |
| GRU   | 7.35          | 0.38               |

GRU, LSTM'e göre hem daha isabetli hem daha hızlı — referans makaledeki
bulguyla tutarlı. **Ancak naive baseline (yarının fiyatı = bugünün fiyatı
kabul edilir) her iki modeli de geçiyor.** Bu, finansal zaman serilerinin
rastgele yürüyüşe yakın olmasından kaynaklanan, literatürde bilinen bir
sonuçtur (bkz. Kaynaklar) — karmaşık modellerin basit bir kıyaslamayı
her zaman geçemeyebileceğini gösteriyor. Detaylı tartışma için
[`docs/approach.md`](docs/approach.md). Gerçek vs tahmin grafiği:

![Gerçek vs Tahmin](docs/lstm_vs_gru_predictions.png)

Tam deney matrisi (6 kombinasyon, dropout testi dahil): [`docs/experiment_results.csv`](docs/experiment_results.csv)

### Tekrarlanabilirlik: 5 farklı rastgele tohum

Sinir ağı ağırlıkları rastgele başlatılır; tek bir koşuda GRU'nun kazanması o
başlangıcın şansı olabilir. Aynı deney (lookback=10, 100 epoch) **beş farklı
tohumla** tekrarlandı:

| Model | Ortalama RMSE ($) | Std sapma | En iyi | En kötü | Ort. süre (s) |
|-------|-------------------|-----------|--------|---------|----------------|
| Naive Baseline | **5,97** | — (deterministik) | — | — | — |
| GRU   | 7,38 | **0,15** | 7,19 | 7,65 | 0,38 |
| LSTM  | 10,26 | 1,38 | 8,71 | 11,94 | 0,51 |

**GRU beş tohumun beşinde de LSTM'i geçti.** Sonuç rastgele başlangıcın şansı
değil.

İkinci ve beklenmedik bulgu: **GRU aynı zamanda çok daha kararlı.** Standart
sapması 0,15; LSTM'inki 1,38 — yaklaşık dokuz kat fark. LSTM'in sonucu hangi
rastgele başlangıçla eğitildiğine belirgin şekilde bağlı (8,71 ile 11,94
arasında değişiyor), GRU ise her koşuda neredeyse aynı yere geliyor. Bu, tek
koşulu bir deneyde görülemeyecek bir sonuçtur.

Naive baseline (5,97 $) GRU'nun beş koşuluk ortalamasından **%24 daha iyi**
kaldı; yani baseline'ın üstünlüğü de tek koşuluk bir tesadüf değil.

Koşu kayıtları: [`docs/tohum_sonuclari.csv`](docs/tohum_sonuclari.csv) ·
Script: `python3 src/tohum_deneyi.py`

## İş Akışı

Proje 10 aşamalı, Ar-Ge sprintleri içeren bir yol haritasıyla geliştirildi
(kurulum → kavramsal temel → literatür araştırması → PyTorch → veri
pipeline → model → eğitim/deney → değerlendirme → dokümantasyon → kritik
değerlendirme). Her aşama ayrı bir Git commit/branch olarak izlenebilir.

## Kaynaklar

**Resmi dokümantasyon:**
- [PyTorch Docs](https://docs.pytorch.org)
- [yfinance Docs](https://ranaroussi.github.io/yfinance/)

**Akademik referanslar:**
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
- Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. EMNLP.
- *Hyperparameter-Optimized RNN, LSTM, and GRU Models for Stock Price Prediction*. MDPI Symmetry (2025).
- Narayanan, A., & Kapoor, S. (2024). *AI Snake Oil: What Artificial Intelligence Can Do, What It Can't, and How to Tell the Difference*. Princeton University Press.

**Uygulama referansı:** Rodolfo Saldanha, [Stock Price Prediction with PyTorch](https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632) (Medium).

## Sınırlılıklar

Hisse senedi fiyat tahmini gerçek dünyada güvenilir bir problem değildir;
bu proje tek hisse/tek özellik (Close fiyatı) ile eğitim amaçlı bir
uygulamadır, bir yatırım aracı değildir. Detaylı kritik değerlendirme
için `docs/approach.md`'ye bakınız.
