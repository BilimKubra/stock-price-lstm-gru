# Yaklaşım Notları (Ar-Ge)

## Veri Kaynağı
- Kaynak: yfinance (canlı API)
- Hisse: AMZN
- Özellik: Close (kapanış fiyatı)
- Neden: Statik CSV yerine canlı veri çekimi hem daha hızlı kurulum hem de esneklik sağlıyor.

## Mimari Kararları (başlangıç noktası)
- Lookback penceresi: 20 gün
- Hidden size: 32
- Katman sayısı: 2
- Giriş boyutu: 1, çıkış boyutu: 1
- Kayıp fonksiyonu: MSELoss
- Optimizer: Adam
- Epoch: 50-100 (başlangıç)

## Gerekçe
Değerler referans PyTorch LSTM/GRU makalesindeki tipik aralıklarla uyumlu;
Aşama 6'daki deney matrisinde (lookback 10/20/30, hidden 16/32/64) sistematik
olarak test edilip gerekirse güncellenecek.
