import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.logistic_regression import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("--- MULAI PROSES PELATIHAN MODEL LOGISTIC REGRESSION ---")

# Load Dataset
df = pd.read_csv('datasetberita.csv')
df = df.dropna(subset=['Clean Narasi'])

X = df['Clean Narasi']
y = df['hoax']
