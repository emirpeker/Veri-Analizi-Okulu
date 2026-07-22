# Veri Analizi ve Makine Öğrenmesi Çalışmaları

Bu repository, **Veri Analizi Okulu** kapsamında gerçekleştirilen temel ve ileri seviye Makine Öğrenmesi (ML) uygulamalarını, veri ön işleme adımlarını ve algoritmik karşılaştırmaları içermektedir. 

Projelerde sadece teorik algoritmalar uygulanmamış; aynı zamanda veri sızıntısı (**data leakage**) ve model-veri tipi uyumsuzluğu gibi pratik mühendislik problemleri tespit edilerek refactor edilmiştir.

---

## 🛠 Proje İçeriği ve Notebook'lar

| Notebook / Konu | Tür | Kullanılan Algoritmalar / Teknikler | Açıklama |
| :--- | :--- | :--- | :--- |
| **Naive Bayes & Random Forest** | Sınıflandırma | `GaussianNB`, `RandomForestClassifier`, `ADASYN` | Akciğer kanseri tahmini. Veri dengesizliği giderildi ve model performansları kıyaslandı. |
| **SVMs (Support Vector Machines)** | Sınıflandırma | `SVC` (Linear, RBF Kernel) | Karmaşık karar sınırlarının analizi ve hiperparametre optimizasyonu. |
| **PCA (Principal Component Analysis)** | Boyut İndirgeme | `PCA` | Yüksek boyutlu veride varyansı koruyarak öznitelik alanını küçültme (Unsupervised). |
| **LDA (Linear Discriminant Analysis)** | Boyut İndirgeme | `LDA` | Sınıf ayrıştırılabilirliğini (class separability) maksimuma çıkaran boyut indirgeme (Supervised). |
| **Çok Sınıflı Sınıflandırma - YSA** | Derin Öğrenme / ML | `Yapay Sinir Ağları (ANN)`, `Softmax` | Çok sınıflı hedef değişkenlerin Yapay Sinir Ağları ile modellenmesi. |
| **K-Ortalamalar (K-Means)** | Kümeleme | `KMeans`, `Elbow Method` | Etiketsiz veride küme merkezlerinin tespiti ve segmented veri analizi. |

---

## 🔍 Algoritma Karşılaştırma Matrisi

| Algoritma | Öğrenme Türü | Temel Kullanım Senaryosu | Kritik Avantajı | Dikkat Edilmesi Gereken Risk / Sınır |
| :--- | :--- | :--- | :--- | :--- |
| **PCA** | Unsupervised | Boyut İndirgeme | Varyansı maksimum tutarak veri boyutunu düşürür. | Dönüştürülen bileşenlerin (components) iş alanındaki anlamlandırılabilirliği kaybolur. |
| **LDA** | Supervised | Boyut İndirgeme & Sınıflandırma | Sınıf etiketlerini kullanarak sınıflar arası mesafeyi max yapar. | Hedef etiketlere (`y`) bağımlıdır, veri etiketsizse kullanılamaz. |
| **SVM** | Supervised | Karmaşık Sınıflandırma | Kernel trick ile doğrusal olmayan verilerde yüksek başarım gösterir. | Büyük ölçekli veri setlerinde bellek ve hesaplama maliyeti yüksektir. |
| **K-Means** | Unsupervised | Müşteri/Veri Segmentasyonu | Hızlıdır, büyük veride ölçeklenmesi kolaydır. | Başlangıç merkezlerine ($k$) hassastır, küremsi olmayan kümelerde zayıftır. |
| **Naive Bayes** | Supervised | Metin / Hızlı Sınıflandırma | Düşük veri miktarıyla hızlı eğitilir, olasılıksal bazlıdır. | Özniteliklerin birbirinden bağımsız olduğu varsayımına (`Naive`) dayanır. |

---

## 📌 Metodolojik Notlar ve Refactoring (Öğrenilen Dersler)

Proje süreçlerinde açık kaynak kodlar ve literatür incelenirken yapılan teknik tespitler:

1. **Data Leakage (Veri Sızıntısı) Düzeltmesi:** 
   * Sentetik veri üretme yöntemleri (`ADASYN` / `SMOTE`), `train_test_split` işleminden **önce** uygulandığında test seti bilgisi eğitim setine sızmaktadır.
   * **Çözüm:** Veri seti önce $X_{train}$ ve $X_{test}$ olarak ayrılmış, oversampling işlemi **yalnızca $X_{train}$** üzerinde uygulanarak test seti tamamen izole edilmiştir.

2. **Model ve Öznitelik Uyumsuzluğu:**
   * Sürekli (continuous) ve ondalıklı değer içeren klinik/medikal verilerde kesikli frekanslar için tasarlanmış `MultinomialNB` yerine sürekli dağılımlara uygun olan `GaussianNB` tercih edilmiştir.

---
