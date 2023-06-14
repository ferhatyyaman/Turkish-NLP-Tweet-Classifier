import pandas as pd
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

# Veri kümesini oku
data = pd.read_csv('veri_kumesi.csv')

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

# Veri kümesini %70 eğitim, %30 test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X_selected, data['label'], train_size=0.7, random_state=42)

# k-NN sınıflandırıcıyı eğitme (k=10, mesafe ölçütü olarak kosinüs benzerliği)
knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
knn.fit(X_train, y_train)

# Test verisi üzerinde tahminler yapma
y_pred = knn.predict(X_test)

# Başarı ve F1 skorlarını hesaplama
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Sonuçları yazdırma
print("Algoritmanın İsmi | Accuracy | F1 score")
print(f"k-NN (k=10, cosine) | {accuracy:.4f} | {f1:.4f}")



# 1. Matris: Ayıklanmış tüm kelimeler ve tweetler
word_index = vectorizer.get_feature_names_out()
word_counts = X.toarray()

df1 = pd.DataFrame(word_counts, columns=word_index)

for word in word_index:
    df1[word] = df1[word] * data['tweet'].apply(lambda x: x.count(word))

df1.drop('tweet', axis=1, inplace=True)
df1.insert(0, 'tweet', data['tweet'])
df1.to_excel('1_matris.xlsx', index=False)


# 2. Matris: Information Gain ile seçilmiş 1000 kelime ve tweetler
selected_index = np.array(vectorizer.get_feature_names_out())[selector.get_support()]
selected_word_counts = X_selected.toarray()

df2 = pd.DataFrame(selected_word_counts, columns=selected_index)

for word in selected_index:
    df2[word] = df2[word] * data['tweet'].apply(lambda x: x.count(word))

df2.drop('tweet', axis=1, inplace=True)
df2.insert(0, 'tweet', data['tweet'])
df2.to_excel('2_matris.xlsx', index=False)
