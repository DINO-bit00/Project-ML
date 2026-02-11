import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
import re
import gradio as gr
import os
from collections import Counter

# --- 1. Definisi Konstanta dan Pra-pemrosesan Teks ---

# Custom Stopwords (sesuai alur RapidMiner)
custom_stopwords_list = [
    'yang', 'dan', 'di', 'ke', 'dari', 'itu', 'ini', 'untuk',
    'pada', 'dengan', 'sebagai', 'atau', 'karena', 'tidak', 'ada'
]
STOPWORDS = set([word.lower() for word in custom_stopwords_list])


# Fungsi Tokenizer Kustom (mereplikasi Process Documents)
def custom_tokenizer(text):
    """
    Menggabungkan Tokenize, Transform Cases, dan Filter Tokens by Length.
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Menggunakan NLTK word_tokenize, yang membutuhkan 'punkt'
    tokens = word_tokenize(text)

    final_tokens = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if 3 <= len(token) <= 30:  # Filter Tokens by Length
            final_tokens.append(token)

    return final_tokens


# --- 2. Fungsi Pelatihan Model (Diperbarui untuk Statistik dan Metrik) ---

def train_model(file_path='datasetberita.csv'):
    """Memuat data, melatih model, dan mengembalikan vektorizer, model, dan metrik."""

    # Memastikan NLTK dapat menemukan punkt
    try:
        nltk.data.path.append(os.getcwd())
        nltk.data.find('tokenizers/punkt')
    except (nltk.downloader.DownloadError, LookupError):
        print("MENGUNDUH NLTK PUNKT: Resource 'punkt' tidak ditemukan. Mengunduh ke direktori proyek...")
        # Unduh ke direktori kerja saat ini
        nltk.download('punkt', download_dir=os.getcwd())

        # Lanjutkan pemrosesan data
    try:
        # 1. Read Excel & Filter Examples
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['hoax', 'Clean Narasi'])

        # 2. Nominal to Text & Menyiapkan Data
        df['Clean Narasi'] = df['Clean Narasi'].astype(str)
        X_raw = df['Clean Narasi']
        y = df['hoax']

        # 3. Split Validation (90% Training, 10% Testing)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.1, random_state=42, stratify=y
        )

        # 4. Process Documents (Vectorization)
        vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
        X_train_vectorized = vectorizer.fit_transform(X_train_raw)

        # 5. Naive Bayes (MultinomialNB dengan Laplace Correction)
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train_vectorized, y_train)

        # 6. Hitung Metrik dan Statistik
        X_test_vectorized = vectorizer.transform(X_test_raw)
        y_pred = model.predict(X_test_vectorized)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Statistik Dataset Penuh
        total_data = len(df)
        count_label = Counter(df['hoax'])
        stat = {
            "Total Data": total_data,
            "Non Hoax (0)": count_label.get(0, 0),
            "Hoax (1)": count_label.get(1, 0)
        }

        # Metrik Pengujian (dari 10% data testing)
        metrik = {
            "Akurasi": f"{report['accuracy'] * 100:.2f}%",
            "Presisi Hoax (1)": f"{report['1']['precision'] * 100:.2f}%",
            "Recall Hoax (1)": f"{report['1']['recall'] * 100:.2f}%",
            "F1-Score Hoax (1)": f"{report['1']['f1-score'] * 100:.2f}%",
        }

        return model, vectorizer, stat, metrik

    except FileNotFoundError:
        print("ERROR: Pastikan file 'data.xlsx' ada di direktori yang sama.")
        return None, None, None, None
    except Exception as e:
        print(f"ERROR: Terjadi kesalahan saat memproses data: {e}")
        return None, None, None, None


# Muat model saat skrip dimulai
global_model, global_vectorizer, global_stat, global_metrik = train_model()


# --- 3. Fungsi Prediksi Gradio (Diperbaiki Output) ---

def predict_hoax(text_input):
    """
    Menerima teks baru dan mengembalikan tiga string terpisah.
    """
    # ... Penanganan error jika model gagal dimuat ...
    if global_model is None or global_vectorizer is None:
        return "Model Gagal Dimuat. Cek Konsol PyCharm.", "N/A", "N/A"

    if not text_input or len(text_input.strip()) < 10:
        return "Masukkan narasi berita yang valid (min 10 karakter).", "N/A", "N/A"

    # Transformasi dan Prediksi
    new_text_vectorized = global_vectorizer.transform([text_input])
    prediction_result = global_model.predict(new_text_vectorized)[0]
    prediction_proba = global_model.predict_proba(new_text_vectorized)[0]

    # Format Hasil
    label = "HOAX" if prediction_result == 1 else "BUKAN HOAX"
    proba_bukan_hoax = f"{prediction_proba[0] * 100:.2f}%"
    proba_hoax = f"{prediction_proba[1] * 100:.2f}%"

    # Mengembalikan tiga nilai terpisah
    return label, proba_hoax, proba_bukan_hoax


# --- 4. Antarmuka Gradio ---

if global_model and global_vectorizer:
    with gr.Blocks(title="Prediksi Hoax Naive Bayes") as demo:
        gr.Markdown("# 📰 Prediksi Berita Hoax (Naive Bayes)")
        # BAGIAN STATISTIK & METRIK
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Statistik Data Awal")
                gr.Textbox(label="Total Data Bersih", value=global_stat['Total Data'], interactive=False)
                gr.Textbox(label="Jumlah BUKAN HOAX (0)", value=global_stat['Non Hoax (0)'], interactive=False)
                gr.Textbox(label="Jumlah HOAX (1)", value=global_stat['Hoax (1)'], interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("## 📈 Hasil Performance (Data Testing)")
                gr.Textbox(label="Akurasi Total", value=global_metrik['Akurasi'], interactive=False)
                gr.Textbox(label="Presisi Kelas HOAX (1)", value=global_metrik['Presisi Hoax (1)'], interactive=False)
                gr.Textbox(label="Recall Kelas HOAX (1)", value=global_metrik['Recall Hoax (1)'], interactive=False)
                gr.Textbox(label="F1-Score Kelas HOAX (1)", value=global_metrik['F1-Score Hoax (1)'], interactive=False)

        gr.Markdown("---")
        gr.Markdown("## Masukkan Teks Berita Baru untuk Prediksi")

        # BAGIAN PREDIKSI
        input_text = gr.Textbox(lines=5, placeholder="Tempelkan narasi 'Clean Narasi' dari berita di sini...",
                                label="Teks Narasi Berita")
        predict_button = gr.Button("Prediksi", variant="primary")

        with gr.Row():
            output_label = gr.Textbox(label="Hasil Prediksi", type="text", interactive=False, container=False)
            output_proba_hoax = gr.Textbox(label="Probabilitas HOAX", type="text", interactive=False)
            output_proba_bukan_hoax = gr.Textbox(label="Probabilitas BUKAN HOAX", type="text", interactive=False)

        # Mengaitkan tombol dengan fungsi
        predict_button.click(
            fn=predict_hoax,
            inputs=input_text,
            outputs=[output_label, output_proba_hoax, output_proba_bukan_hoax]
        )

    demo.launch()
else:
    print(
        "\n[LAUNCH FAILED] Aplikasi Gradio tidak dapat diluncurkan karena gagal memuat model. Cek error konsol di atas dan pastikan 'data.xlsx' tersedia dan benar.")