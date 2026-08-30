## Proje Asamalari ve Commit Karsiliklari

| Asama | Aciklama | Commit(ler) |
|-------|----------|-------------|
| 0 | Ortam kurulumu, klasor iskeleti | `6beac3a`, `44e87f5` |
| 1 | Kavramsal temel (ML fundamentals) | - (kod uretmedi, sohbette islendi) |
| 2 | Ar-Ge sprint 1: veri kaynagi + mimari kararlari | `846aa76`, `a35607a`, `81d2cb6` |
| 3 | PyTorch mekaniginin gercek veriyle dogrulanmasi | `987b629` |
| 4 | Veri pipeline (normalize, sliding window, split) | `bdef219` |
| 5 | LSTM/GRU model siniflari | `bf3c3e8` |
| 6 | Ar-Ge sprint 2: deney matrisi + dropout testi | `1b5bac5`, `c979637` |
| 7 | Final degerlendirme | `4c14e22` |
| 8 | README ve dokumantasyon | `84a4fe7` |
| 9 | Kritik degerlendirme ve kapanis | `8924d4d` |
| 10 | Jupyter Notebook | `02c1928` |
| 11 | Naive baseline + testler | `00b1490` |

---

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

## Asama 9: Kritik Degerlendirme ve Kapanis

### Sonuclarin Sinirlari
- Bu proje tek bir hisseye (AMZN), tek bir ozellige (Close fiyati) ve
  ~2 yillik gunluk veriye dayaniyor. Sonuclar (GRU RMSE=7.35$, LSTM
  RMSE=11.94$) bu dar kapsamda gecerli; farkli bir hisse, zaman araligi
  veya piyasa kosulunda ayni siralama (GRU > LSTM) garanti degil.
- Asama 6'daki deneyde gozlemlenen train/test loss farki, klasik
  overfitting'den cok train ve test donemleri arasindaki piyasa rejimi
  farkindan (distribution shift) kaynaklaniyor olabilir - hisse
  fiyatlari zaman icinde istatistiksel olarak durgun (stationary)
  degildir, bu da zaman serisi tahmininin temel zorluklarindan biridir.
- Model sadece fiyat gecmisini kullaniyor; hacim, haber/sentiment,
  makroekonomik veri gibi gercek fiyat hareketini etkileyen faktorler
  disarida birakildi.
- Akademik/pratik konsensus: hisse fiyati tahmini, kisa vadeli
  gurultunun (noise) sinyalden ayirt edilmesinin cok zor oldugu,
  bilinen sekilde guvenilmez bir problemdir. Bu proje bir yatirim
  araci degil, bir ogrenme/portfoy calismasidir.

### Elestirel Okuma Notu
Narayanan, A., & Kapoor, S. (2024). *AI Snake Oil: What Artificial
Intelligence Can Do, What It Can't, and How to Tell the Difference*.
Princeton University Press. Bu kitap, ML'in hype ile gercek performansi
arasindaki farki ayirt etmenin onemini vurguluyor - hisse tahmini gibi
kaotik, dusuk sinyal-gurultu oranli problemlerde bu ayrim ozellikle
kritik.

### Sonraki Adimlar (bonus, kapsam disi birakildi)
- Ek ozellikler: islem hacmi, teknik indikatorler (RSI, MACD)
- Coklu hisse / sektor bazli genelleme testi
- Transformer tabanli zaman serisi modelleri (ör. Temporal Fusion
  Transformer) ile karsilastirma
- Daha uzun tarihsel veri (5-10 yil) ile tekrar deney

## Asama 11: Naive Baseline Karsilastirmasi

Ar-Ge sonuclarinin bilimsel olarak degerlendirilebilmesi icin bir naive baseline eklendi: "yarinin fiyati = bugunun fiyati" (persistence/random walk varsayimi).

**Sonuc:** Naive baseline test RMSE = $5.97, hem LSTM'i ($11.94) hem de GRU'yu ($7.35) geciyor.

**Yorum:** Bu sonuc finansal zaman serilerinin (ozellikle gunluk kapanis fiyatlarinin) rastgele yuruyuse (random walk) yakin olmasindan kaynaklanan, literaturde bilinen bir olgudur. Karmasik sinir agi mimarilerinin (LSTM, GRU) bu problemde ozellikle avantaj saglamadigini, aksine ekstra karmasiklik/hesaplama maliyeti getirdigini gosteriyor. Bu, Narayanan & Kapoor (2024) "AI Snake Oil" kitabinin merkezi elestirisiyle dogrudan ortusuyor: ML modelleri, basit bir kiyaslamayla karsilastirilmadan degerlendirildiginde yanlis bir basari izlenimi verebiliyor.

**Sonuc:** Bu proje icin LSTM/GRU karsilastirmasi hala pedagojik olarak degerli (mimari farkliliklarini, egitim surecini, PyTorch kullanimini ogretiyor), ancak "GRU en iyi model" gibi bir iddia bu baseline olmadan eksik olurdu.
