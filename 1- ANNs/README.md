## ANN Çıktısı

Aşağıda, ANNs çalışmasının bir görsel çıktısı bulunmaktadır:

![ANN Output](https://raw.githubusercontent.com/mustafaemirata/PyTorch/main/1-%20ANNs/ann_output.png)

---

## Proje Hakkında

Bu proje, PyTorch kullanılarak basit bir yapay sinir ağı (ANN) modeliyle MNIST el yazısı rakam veri seti üzerinde sınıflandırma problemi çözmeyi amaçlamaktadır.

- **Veri Seti:** MNIST el yazısı rakam veri seti, 28x28 boyutunda gri tonlamalı görüntülerden oluşur.  
- **Model:** Çok katmanlı yapay sinir ağı (Multi-layer Perceptron)  
  - Giriş katmanı: 28x28 boyutlu görüntü, tek boyutlu vektöre dönüştürülür  
  - İki gizli katman: 128 ve 64 nöronlu, ReLU aktivasyonlu  
  - Çıktı katmanı: 10 sınıf (rakamlar 0-9)  
- **Eğitim:**  
  - Adam optimizasyon algoritması  
  - Cross-Entropy kayıp fonksiyonu  
  - Batch size: 64  
  - Epoch sayısı: 5  
- **Amaç:** MNIST veri setinde yüksek doğrulukla sınıflandırma yapmak.
