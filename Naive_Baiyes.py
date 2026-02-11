import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("--- MULAI PROSES PELATIHAN MODEL ---")

# Load Dataset
df = pd.read_csv('datasetberita.csv')
df = df.dropna(subset=['Clean Narasi'])
X = df['Clean Narasi']
y = df['hoax']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('nb', MultinomialNB())
])

print("Sedang melatih model...")
model_pipeline.fit(X_train, y_train)
print("Pelatihan selesai!")

# Evaluasi
y_pred = model_pipeline.predict(X_test)
print("\n--- LAPORAN PERFORMA ---")
print(f"Akurasi Final: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Fakta', 'Hoax']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Prediksi Fakta','Prediksi Hoax'],
            yticklabels=['Asli Fakta','Asli Hoax'])
plt.savefig('final_confusion_matrix.png')
plt.show()

# Wordcloud
plt.figure(figsize=(15, 7))
subset_fakta = df[df['hoax'] == 0]['Clean Narasi']
subset_hoax = df[df['hoax'] == 1]['Clean Narasi']

wc_fakta = WordCloud(width=800, height=400).generate(" ".join(subset_fakta))
wc_hoax = WordCloud(width=800, height=400).generate(" ".join(subset_hoax))

plt.subplot(1, 2, 1)
plt.imshow(wc_fakta)
plt.title("Wordcloud Berita FAKTA", fontsize=16)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(wc_hoax)
plt.title("Wordcloud Berita HOAX", fontsize=16)
plt.axis('off')

plt.savefig('final_wordcloud.png')
plt.show()

# SIMPAN MODEL
joblib.dump(model_pipeline, 'model_hoax_complete.pkl')
print("\nModel berhasil disimpan sebagai 'model_hoax_complete.pkl'. Siap digunakan!")
