import pandas as pd
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from gensim.models import Word2Vec
import numpy as np

# Veri kümesini oku
data = pd.read_csv('Nefret_Söylemi_veri_kümesi_10k.csv', na_values=[''])
data = data.dropna(subset=['tweet'])
data['tweet'].fillna('', inplace=True)

# Zemberek kütüphanesini yükle ve JVM'i başlat
jar_path = r'C:\Users\ferhat\zemberek-nlp\all\target\zemberek-full.jar'
startJVM(getDefaultJVMPath(), '-ea', f'-Djava.class.path={jar_path}')

TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()

# Her tweet için kelimeleri ve köklerini bul
stems_list = []
for tweet in data['tweet']:
    words = tweet.split()
    stems = []
    for word in words:
        analysis = morphology.analyzeAndDisambiguate(JString(word))
        results = analysis.bestAnalysis()
        for result in results:
            stems.append(str(result.getStem()))

    stems_list.append(' '.join(stems))

# JVM'i kapat
shutdownJVM()

# Kelime-frekans vektörü oluşturma
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(stems_list)

# Information Gain ile en önemli 1000 kelimeyi seçme
selector = SelectKBest(mutual_info_classif, k=1000)
X_selected = selector.fit_transform(X, data['label'])

# TF-IDF dönüşümü
transformer = TfidfTransformer()
X_tfidf = transformer.fit_transform(X_selected)

# Veri kümesini %70 eğitim, %30 test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['label'], train_size=0.7, random_state=42)

# SVM sınıflandırıcıyı eğitme
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Word2Vec-CBOW modelini eğitme
word2vec_model = Word2Vec(stems_list, sg=0, window=5, min_count=1)

# Word2Vec-CBOW ile vektörleri temsil etme
word2vec_vectors = []
for tweet in stems_list:
    vector = []
    for word in tweet.split():
        if word in word2vec_model.wv:
            vector.append(word2vec_model.wv[word])
    if len(vector) > 0:
        word2vec_vectors.append(np.mean(vector, axis=0))
    else:
        word2vec_vectors.append(np.zeros(100))  # Eğer vektör oluşturulamazsa sıfır vektörü kullan

# Word2Vec-CBOW vektörlerini ve etiketleri yeniden boyutlandırma
word2vec_vectors = np.array(word2vec_vectors)
y_data = np.array(data['label'])
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(
    word2vec_vectors, y_data, train_size=0.7, random_state=42)

# SVM sınıflandırıcısını Word2Vec-CBOW ile eğitme
svm_word2vec = SVC()
svm_word2vec.fit(X_train_word2vec, y_train_word2vec)
y_pred_svm_word2vec = svm_word2vec.predict(X_test_word2vec)

# Decision Tree sınıflandırıcısını eğitme
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Başarı ve F1 skorlarını hesaplama
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

accuracy_svm_word2vec = accuracy_score(y_test_word2vec, y_pred_svm_word2vec)
f1_svm_word2vec = f1_score(y_test_word2vec, y_pred_svm_word2vec, average='weighted')

# Sonuçları yazdırma
print("Algoritmanın İsmi | Accuracy | F1 score")
print(f"SVM | {accuracy_svm:.4f} | {f1_svm:.4f}")
print(f"Decision Tree | {accuracy_dt:.4f} | {f1_dt:.4f}")
print(f"SVM (Word2Vec-CBOW) | {accuracy_svm_word2vec:.4f} | {f1_svm_word2vec:.4f}")
