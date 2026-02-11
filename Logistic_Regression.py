import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("--- MULAI PROSES PELATIHAN MODEL LOGISTIC REGRESSION ---")

# Load Dataset
df = pd.read_csv('datasetberita.csv')
df = df.dropna(subset=['Clean Narasi'])

X = df['Clean Narasi']
y = df['hoax']

# Pipleine
pipeline = Pipeline([
	('tfidf', TfidfVectorizer()),
	('classifier', LogisticRegression())
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
