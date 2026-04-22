# 🧠 Çok Modelli MRI Beyin Tümörü Sınıflandırması

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn)
![Lisans](https://img.shields.io/badge/Lisans-MIT-green?style=for-the-badge)

**Gelişmiş ön işleme, çok modelli eğitim ve karşılaştırmalı değerlendirme içeren kapsamlı bir derin öğrenme tabanlı MRI beyin tümörü sınıflandırma pipeline'ı.**

</div>

---

## 📋 İçindekiler

- [Genel Bakış](#-genel-bakış)
- [Özellikler](#-özellikler)
- [Mimari](#-mimari)
- [Ön İşleme Pipeline'ı](#-ön-işleme-pipelineı)
- [Desteklenen Modeller](#-desteklenen-modeller)
- [Kurulum](#-kurulum)
- [Veri Seti Yapısı](#-veri-seti-yapısı)
- [Yapılandırma](#-yapılandırma)
- [Kullanım](#-kullanım)
- [Sonuçlar ve Değerlendirme](#-sonuçlar-ve-değerlendirme)
- [Görselleştirmeler](#-görselleştirmeler)
- [Proje Yapısı](#-proje-yapısı)
- [Katkıda Bulunma](#-katkıda-bulunma)

---

## 🔍 Genel Bakış

Bu proje, birden fazla son teknoloji evrişimli sinir ağı kullanarak **MRI beyin tümörü sınıflandırması** için uçtan uca eksiksiz bir pipeline sunmaktadır. Pipeline şunları kapsar:

- MRI'ya özgü gelişmiş görüntü ön işleme
- Yapılandırılabilir veri artırma (augmentation)
- Çok modelli eğitim ve değerlendirme
- Zengin görselleştirmelerle karşılaştırmalı performans analizi

Sistem modüler, tekrarlanabilir ve genişletilebilir biçimde tasarlanmıştır; hem araştırma hem de klinik karar destek uygulamaları için uygundur.

---

## ✨ Özellikler

| Özellik | Açıklama |
|---|---|
| 🔬 **MRI Ön İşleme** | 6 adımlı alana özgü pipeline (arka plan kaldırma, bias düzeltme, gürültü giderme, normalizasyon, CLAHE, keskinleştirme) |
| 🔁 **Veri Artırma** | Yatay çevirme, döndürme, parlaklık/kontrast değişimi, çok yamalı rastgele silme |
| 🤖 **Çok Model Desteği** | ResNet, EfficientNet, DenseNet, VGG, MobileNet dahil 8 mimari |
| 📊 **Zengin Değerlendirme** | Doğruluk, Hassasiyet, Geri Çağırma, F1, ROC-AUC eğrileri, Karmaşıklık Matrisi |
| 📈 **Karşılaştırma Raporu** | Tüm eğitilmiş modellerin yan yana sıralanması |
| 🖼️ **Görselleştirmeler** | Veri seti dağılımı, ön işleme adımları, artırma örnekleri, eğitim eğrileri |
| ⚙️ **Yapılandırılabilir** | Tüm hiperparametreler ve yollar tek bir `Config` dataclass'ında toplanmış |

---

## 🏗️ Mimari

```
┌─────────────────────────────────────────────────────────┐
│              MRI Sınıflandırma Pipeline'ı               │
├─────────────────────────────────────────────────────────┤
│  Veri Yükleme → Ön İşleme → Artırma → Eğitim           │
│                                                         │
│  ┌──────────────┐    ┌────────────────────────────────┐ │
│  │ MRIDataset   │───▶│      MRIPreprocessor           │ │
│  │ (PyTorch)    │    │  1. Arka Plan Kaldırma         │ │
│  └──────────────┘    │  2. Bias Alan Düzeltme         │ │
│                      │  3. Gelişmiş Gürültü Giderme   │ │
│  ┌──────────────┐    │  4. Yoğunluk Normalizasyonu    │ │
│  │ ModelFactory │    │  5. CLAHE Geliştirme           │ │
│  │  - ResNet    │    │  6. Kenar Keskinleştirme       │ │
│  │  - EfficNet  │    └────────────────────────────────┘ │
│  │  - DenseNet  │                                       │
│  │  - VGG       │    ┌────────────────────────────────┐ │
│  │  - MobileNet │    │      ModelEvaluator            │ │
│  └──────────────┘    │  - Karmaşıklık Matrisi         │ │
│                      │  - ROC / AUC Eğrileri          │ │
│  ┌──────────────┐    │  - Sınıflandırma Raporu        │ │
│  │ ModelTrainer │    └────────────────────────────────┘ │
│  │  - AdamW     │                                       │
│  │  - LR Sched  │    ┌────────────────────────────────┐ │
│  └──────────────┘    │  MultiModelComparison          │ │
│                      │  - Sıralı Rapor                │ │
│                      │  - Çubuk Grafikler             │ │
│                      └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 🔬 Ön İşleme Pipeline'ı

Her MRI görüntüsü modele girmeden önce sıralı 6 adımlı ön işleme pipeline'ından geçirilir:

```
Orijinal Görüntü
       │
       ▼
① Arka Plan Kaldırma       — Otsu eşikleme + morfolojik işlemler + en büyük bağlı bileşen
       │
       ▼
② Bias Alan Düzeltme       — Gaussian yumuşatma tabanlı yoğunluk alanı normalizasyonu
       │
       ▼
③ Gelişmiş Gürültü Giderme — YUV parlaklık kanalında yerel olmayan ortalama gürültü giderme
       │
       ▼
④ Yoğunluk Normalizasyonu  — Beyin maskesi içinde min-maks normalizasyonu
       │
       ▼
⑤ CLAHE Geliştirme         — Sınırlı Uyarlamalı Histogram Eşitleme
       │
       ▼
⑥ Kenar Keskinleştirme     — Yapısal detay için unsharp masking
       │
       ▼
İşlenmiş Görüntü (224×224)
```

---

## 🤖 Desteklenen Modeller

| Model | Mimari | Parametre Sayısı (yaklaşık) |
|---|---|---|
| `resnet-18` | ResNet-18 | ~11M |
| `resnet-50` | ResNet-50 | ~25M |
| `resnet-101` | ResNet-101 | ~45M |
| `efficientnet-b0` | EfficientNet-B0 | ~5M |
| `mobilenet-v2` | MobileNet-V2 | ~3M |
| `densenet-201` | DenseNet-201 | ~20M |
| `vgg-16` | VGG-16 | ~138M |
| `vgg-19` | VGG-19 | ~144M |

Tüm modeller **ImageNet önceden eğitilmiş ağırlıkları** kullanır; düzenlileştirme için `Dropout(0.5)` eklenmiş özel bir sınıflandırma başlığı ile yapılandırılmıştır.

---

## ⚙️ Kurulum

### Gereksinimler

- Python 3.8+
- CUDA destekli GPU (önerilir)

### Adım 1: Depoyu klonlayın

```bash
git clone https://github.com/kullaniciadi/mri-tumor-classification.git
cd mri-tumor-classification
```

### Adım 2: Sanal ortam oluşturun

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Adım 3: Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tqdm>=4.65.0
```

---

## 📁 Veri Seti Yapısı

Pipeline, veri setinin aşağıdaki formatta düzenlenmiş olmasını bekler:

```
archive/
├── training/
│   ├── glioma/
│   │   ├── goruntu001.jpg
│   │   └── ...
│   ├── meningioma/
│   │   └── ...
│   ├── notumor/
│   │   └── ...
│   └── pituitary/
│       └── ...
└── testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

> Bu proje, Kaggle üzerindeki [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) ile uyumludur.

`Config` içindeki yolları kendi kurulumunuza göre güncelleyin (bkz. [Yapılandırma](#-yapılandırma)).

---

## 🛠️ Yapılandırma

Tüm parametreler `main.py` dosyasının üstündeki `Config` dataclass'ında toplanmıştır:

```python
@dataclass
class Config:
    # Yollar
    TRAIN_PATH: str = '/content/drive/MyDrive/archive/training'
    TEST_PATH:  str = '/content/drive/MyDrive/archive/testing'

    # Görüntü
    IMG_SIZE: Tuple[int, int] = (224, 224)

    # Eğitim
    BATCH_SIZE:    int   = 32
    NUM_EPOCHS:    int   = 10
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY:  float = 1e-4

    # Ön İşleme
    GAUSSIAN_SIGMA:    float          = 80.0
    CLAHE_CLIP_LIMIT:  float          = 2.5
    CLAHE_TILE_SIZE:   Tuple[int,int] = (8, 8)
    DENOISE_H:         int            = 5

    # Veri Artırma
    RANDOM_ERASING_PROB: float = 0.2
    ROTATION_RANGE:      Tuple = (-50, 50)
    BRIGHTNESS_RANGE:    Tuple = (0.50, 1.50)
    CONTRAST_RANGE:      Tuple = (0.50, 1.50)

    # Görselleştirme
    NUM_SAMPLE_IMAGES: int  = 3
    VISUALIZE_STEPS:   bool = True
```

---

## 🚀 Kullanım

### Tam pipeline'ı çalıştırın

```bash
python main.py
```

Bu komut sırasıyla şunları yapacaktır:
1. Veri setini yükler ve doğrular
2. Veri seti istatistiklerini gösterir
3. Ön işleme ve artırma görselleştirmelerini üretir
4. Tüm modelleri sırayla eğitir
5. Her modeli test seti üzerinde değerlendirir
6. Karşılaştırmalı performans raporunu yazdırır

### Tek bir modeli programatik olarak eğitin

```python
from main import Config, ModelFactory, ModelTrainer, ModelEvaluator
import torch

config = Config()
model = ModelFactory.create_model('efficientnet-b0', num_classes=4)

trainer = ModelTrainer(model, 'efficientnet-b0', config, train_loader, class_names)
history = trainer.train()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
evaluator = ModelEvaluator(model, 'efficientnet-b0', test_loader, class_names, device)
results = evaluator.evaluate()
```

### Tek bir görüntüyü ön işleyin

```python
import cv2
from main import Config, MRIPreprocessor

config = Config()
preprocessor = MRIPreprocessor(config)

img = cv2.cvtColor(cv2.imread('tarama.jpg'), cv2.COLOR_BGR2RGB)
islenmis, adimlar = preprocessor.process_pipeline(img, return_steps=True)
# `adimlar` her aşamadaki ara sonuçları içerir
```

---

## 📊 Sonuçlar ve Değerlendirme

Eğitim tamamlandıktan sonra her model otomatik olarak değerlendirilir ve sıralı bir karşılaştırma tablosu yazdırılır:

```
Model                Doğruluk     Hassasiyet   Geri Çağırma F1-Skoru
----------------------------------------------------------------------
efficientnet-b0      0.9712       0.9718       0.9712       0.9713
resnet-50            0.9634       0.9641       0.9634       0.9636
densenet-201         0.9598       0.9605       0.9598       0.9600
mobilenet-v2         0.9521       0.9533       0.9521       0.9525
resnet-101           0.9487       0.9496       0.9487       0.9490
```

> ✅ **En İyi Model:** Çıktıda otomatik olarak vurgulanır

### Her model için hesaplanan metrikler

- **Doğruluk (Accuracy)** — genel sınıflandırma başarı oranı
- **Hassasiyet / Geri Çağırma / F1** — tüm sınıflar üzerinde ağırlıklı ortalama
- **ROC-AUC** — sınıf başına bire karşı geri kalan eğrileri
- **Karmaşıklık Matrisi** — ısı haritası görselleştirmesi

---

## 🖼️ Görselleştirmeler

Pipeline aşağıdaki grafikleri otomatik olarak üretir:

| Görselleştirme | Açıklama |
|---|---|
| Veri Seti Dağılımı | Eğitim ve test sınıf sayıları için çubuk grafikler |
| Ön İşleme Adımları | Her örnek için 6 pipeline aşamasının yan yana görünümü |
| Artırma Örnekleri | Her artırma türünü gösteren 9 panelli ızgara |
| Eğitim Eğrileri | Her model için epoch başına kayıp ve doğruluk |
| Karmaşıklık Matrisi | Tahmin ile gerçek etiketlerin ısı haritası |
| ROC Eğrileri | Test seti üzerinde sınıf başına AUC eğrileri |
| Model Karşılaştırması | Tüm metrikler için yatay çubuk grafikler |

---

## 📂 Proje Yapısı

```
mri-tumor-classification/
├── main.py                  # Giriş noktası — tam pipeline
├── requirements.txt         # Python bağımlılıkları
├── README.md
└── modüller (tümü main.py içinde)
    ├── Config               # Merkezi hiperparametreler
    ├── MRIPreprocessor      # 6 adımlı ön işleme pipeline'ı
    ├── MRIDataset           # Artırma destekli PyTorch Dataset
    ├── DataVisualizer       # Görselleştirme yardımcıları
    ├── ModelFactory         # Özel başlıklı model oluşturma
    ├── ModelTrainer         # Eğitim döngüsü + LR zamanlayıcı
    ├── ModelEvaluator       # Çıkarım + metrikler + grafikler
    └── MultiModelComparison # Modeller arası sıralama ve rapor
```

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Katkıda bulunmak için:

1. Depoyu fork'layın
2. Özellik dalı oluşturun: `git checkout -b ozellik/yeni-ozellik`
3. Değişikliklerinizi kaydedin: `git commit -m 'Yeni özellik eklendi'`
4. Dalı gönderin: `git push origin ozellik/yeni-ozellik`
5. Pull Request açın

Kodunuzun mevcut stili izlediğinden ve uygun docstring'ler içerdiğinden emin olun.

---

## 📄 Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır — ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.

---

## 🙏 Teşekkürler

- Derin öğrenme altyapısı için [PyTorch](https://pytorch.org/) ve [torchvision](https://pytorch.org/vision/)
- Görüntü işleme için [OpenCV](https://opencv.org/)
- Değerlendirme metrikleri için [scikit-learn](https://scikit-learn.org/)
- Kaggle üzerindeki [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

<div align="center">
Tıbbi görüntüleme araştırmaları için ❤️ ile yapılmıştır
</div>
