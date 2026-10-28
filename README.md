# Veri Analizi ve Makine Öğrenmesi Çalışmaları

Bu repository, **Veri Analizi Okulu** kapsamında gerçekleştirilen veri analizi ve makine öğrenmesi uygulamalarını içermektedir.

Çalışmalarda sınıflandırma, kümeleme, boyut indirgeme, veri ön işleme ve model değerlendirme yöntemleri Jupyter Notebook ortamında uygulanmıştır.

**Kullanılan teknolojiler:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras ve Imbalanced-learn.

---

## Proje İçeriği ve Notebook'lar

| Notebook / Konu | Tür | Kullanılan Algoritmalar / Teknikler | Açıklama |
| :--- | :--- | :--- | :--- |

| [Naive Bayes & Random Forest](./Veri_Analiz_Okulu%2C_Yapay_Zeka__Makina_%C3%96%C4%9Frenmesi%2C_Ders_8_Uygulama.ipynb) | Sınıflandırma | `MultinomialNB`, `RandomForestClassifier`, `RandomOverSampler` | Akciğer kanseri anket verileri üzerinde sınıflandırma ve sınıf dengesizliği çalışması. |

| Akciğer kanseri anket verileri üzerinde sınıflandırma ve sınıf dengesizliği çalışması. |
| [SVM (Support Vector Machines)](./Veri_Analiz_Okulu%2C_Yapay_Zeka__Makina_%C3%96%C4%9Frenmesi%2C_Ders_6_Uygulama__SVM.ipynb) | Sınıflandırma | `SVC`, Linear ve RBF Kernel | Meme kanseri veri seti üzerinde SVM modellerinin eğitilmesi ve karşılaştırılması. |
| [PCA (Principal Component Analysis)](./Veri_Analiz_Okulu%2C_Yapay_Zeka__Makina_%C3%96%C4%9Frenmesi%2C_Ders_6_Uygulama__PCA.ipynb) | Boyut İndirgeme | `PCA`, `StandardScaler` | Iris veri setinde açıklanan varyansın incelenmesi ve boyut indirgeme. |
| [LDA (Linear Discriminant Analysis)](./Veri_Analiz_Okulu%2C_Yapay_Zeka__Makina_%C3%96%C4%9Frenmesi%2C_Ders_6_Uygulama__LDA.ipynb) | Sınıflandırma / Boyut İndirgeme | `LinearDiscriminantAnalysis` | Palmer Penguins veri seti üzerinde sınıf ayrıştırma ve görselleştirme. |
| [Çok Sınıflı Sınıflandırma – YSA](./Veri_Analiz_Okulu%2C_Yapay_Zeka__Makina_%C3%96%C4%9Frenmesi%2C_Ders_5_Uygulama_Script_1.ipynb) | Makine Öğrenmesi / Derin Öğrenme | Yapay Sinir Ağları (ANN), Softmax | Penguen türlerinin çok sınıflı bir sinir ağı modeliyle sınıflandırılması. |
| [K-Ortalamalar (K-Means)](./Veri%20Analiz%20Okulu%2C%20Yapay%20Zeka_%20Makina%20%C3%96%C4%9Frenmesi%2C%20Ders%205%20Uygulama%20Script%202.ipynb) | Kümeleme | `KMeans`, Elbow Method, Silhouette Score | Sentetik veri kümelerinde kümeleme, uygun küme sayısı ve ölçeklendirme etkilerinin incelenmesi. |

---

## Algoritma Karşılaştırma Matrisi

| Algoritma | Öğrenme Türü | Temel Kullanım Senaryosu | Avantajı | Dikkat Edilmesi Gereken Nokta |
| :--- | :--- | :--- | :--- | :--- |
| **PCA** | Gözetimsiz (Unsupervised) | Boyut İndirgeme | Verideki varyansı koruyan bileşenler oluşturur. | Bileşenlerin yorumlanması zorlaşabilir. |
| **LDA** | Gözetimli (Supervised) | Sınıflandırma ve Boyut İndirgeme | Sınıflar arasındaki ayrımı artırmayı hedefler. | Sınıf etiketlerine ihtiyaç duyar. |
| **SVM** | Gözetimli | Sınıflandırma | Kernel fonksiyonlarıyla doğrusal olmayan ilişkileri modelleyebilir. | Ölçeklendirmeye ve hiperparametre seçimine duyarlıdır. |
| **K-Means** | Gözetimsiz | Kümeleme | Basit ve birçok veri kümesinde verimli bir yöntemdir. | Küme sayısına, ölçeklendirmeye ve küme geometrisine duyarlıdır. |
| **Naive Bayes** | Gözetimli | Sınıflandırma | Hesaplama açısından verimli olasılıksal bir yöntemdir. | Özellikler arasında koşullu bağımsızlık varsayımı yapar. |
| **Random Forest** | Gözetimli | Sınıflandırma | Birden fazla karar ağacının sonuçlarını birleştirir. | Model karmaşıklığı ve yorumlanabilirlik dikkate alınmalıdır. |

---

## Metodolojik Notlar ve Geliştirme Alanları

Bu çalışmalar eğitim amaçlı notebook'lardan oluşmaktadır. İnceleme sırasında bazı teknik iyileştirme alanları belirlenmiştir.


### 1. Data Leakage (Veri Sızıntısı)

Akciğer kanseri sınıflandırma çalışmasında, başlangıçta yeniden örnekleme işlemi eğitim/test ayrımından önce gerçekleştiriliyordu. Bu durum veri sızıntısı riski oluşturuyordu.

**Yapılan düzeltmeler:**

- Veri seti önce eğitim ve test kümelerine ayrıldı.
- Sınıf oranlarını korumak için `stratify=y` kullanıldı.
- Kategorik özelliklere daha uygun olması nedeniyle `ADASYN` yerine `RandomOverSampler` tercih edildi.
- Yeniden örnekleme yalnızca eğitim kümesine uygulandı.
- Modeller değiştirilmemiş test kümesi üzerinde yeniden değerlendirildi.

**Güncel sonuçlar (69 test örneği):**

| Metrik | Naive Bayes | Random Forest |
| --- | ---: | ---: |
| Accuracy | 86% | 88% |
| Class 0 Recall | 80% | 40% |
| Class 1 Recall | 86% | 97% |
| Macro F1-score | 0.76 | 0.72 |

Bu sonuçlar tek bir eğitim/test ayrımına dayanmaktadır. Özellikle küçük ve dengesiz test kümesi nedeniyle genellenebilir performansın belirlenmesi için ek değerlendirmeler gereklidir.


### 2. Naive Bayes Model Seçimi

Akciğer kanseri notebook'unda `MultinomialNB` kullanılmaktadır.

Bu algoritma, negatif olmayan özelliklerle çalışır ve özellikle sayım/frekans türündeki veriler için geliştirilmiştir. Veri setinin özellik dağılımlarına göre `GaussianNB` gibi alternatiflerin karşılaştırılması değerlendirilmesi gereken bir geliştirme alanıdır.

### 3. Model Değerlendirme

Sınıflandırma çalışmalarında doğruluk (accuracy), karmaşıklık matrisi (confusion matrix), precision, recall ve F1-score gibi ölçütler incelenmektedir.

Özellikle dengesiz veri setlerinde tek başına accuracy değerine güvenmek yeterli değildir.


---

## Çalıştırma

Notebook'lar Google Colab veya uygun Python ortamına sahip Jupyter Notebook üzerinde incelenebilir.

Kullanılan başlıca Python paketleri:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `imbalanced-learn`

Gerekli paketler ve veri kaynakları notebook'a göre değişebilir.


**Veri seti:** Akciğer kanseri sınıflandırma notebook'u CSV verisini Hugging Face üzerindeki `nateraw/lung-cancer` veri kaynağından yüklemektedir. Çalıştırma sırasında internet bağlantısı gereklidir.


---

## Proje Kapsamı

Bu repository, makine öğrenmesi algoritmalarını öğrenmek, uygulamak ve sonuçlarını karşılaştırmak amacıyla hazırlanmış **eğitim çalışmalarından** oluşmaktadır.

Notebook'larda kullanılan örnekler ve veri kaynakları ilgili çalışmaların bağlamında değerlendirilmelidir. Bunlar üretim ortamında kullanılan veya klinik karar verme amacı taşıyan sistemler değildir.
