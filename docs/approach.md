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

## Asama 6 Ar-Ge Sonucu
- Deney matrisi (lookback 10/20/30 x LSTM/GRU): GRU her lookback degerinde LSTM'den
  daha dusuk test loss ve daha kisa egitim suresi verdi (tutarli sekilde).
- En iyi konfigurasyon: GRU, lookback=10, hidden=32, test_loss=0.0151
- Dropout denemesi (0.3): train VE test loss'u kotulestirdi (0.0151 -> 0.0164).
  Sebep muhtemelen klasik overfitting degil, train/test donemleri arasindaki
  piyasa rejimi farkindan (distribution shift) kaynaklaniyor - dropout bu
  sorunu cozmuyor. Final model dropout'suz (0.0) birakildi.
- Final karar: GRU, lookback=10, hidden=32, dropout=0.0
