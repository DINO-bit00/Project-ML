import gradio as gr
import joblib
import os


# 1. FUNGSI LOAD MODEL
def load_model():
    """Memuat model dari file .pkl"""
    path = 'model_hoax_complete.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan. Pastikan file ada di folder yang sama.")
    return joblib.load(path)


# Load model di awal agar tidak berat saat prediksi
try:
    model = load_model()
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error: {e}")
    model = None


# 2. FUNGSI PREDIKSI
def deteksi_hoax(text_input):
    """
    Fungsi ini menerima teks, melakukan prediksi,
    dan mengembalikan probabilitas untuk Gradio.
    """
    if model is None:
        return "Model tidak ditemukan."

    if not text_input.strip():
        return "Mohon masukkan teks berita."

    # Prediksi Probabilitas (Fakta vs Hoax)
    proba = model.predict_proba([text_input])[0]

    # Return dalam format Dictionary agar Gradio membuatkan grafik bar
    # Asumsi: Index 0 = Fakta, Index 1 = Hoax
    return {
        "FAKTA (Valid)": float(proba[0]),
        "HOAX (Palsu)": float(proba[1])
    }


# 3. ANTARMUKA GRADIO (UI)
# Contoh berita untuk demo cepat
contoh_berita = [
    ["Vaksin COVID-19 mengandung microchip magnetik yang ditanamkan oleh elit global untuk melacak manusia."],
    ["Presiden Joko Widodo meresmikan jalan tol baru di Sumatera yang menghubungkan dua provinsi besar."],
    ["Beredar kabar bahwa meminum air garam hangat dapat membunuh virus corona seketika di tenggorokan."]
]

# Membangun Interface
demo = gr.Interface(
    fn=deteksi_hoax,  # Fungsi otak utamanya
    inputs=gr.Textbox(lines=5, placeholder="Tempel narasi berita di sini...", label="Input Berita"),
    outputs=gr.Label(num_top_classes=2, label="Hasil Analisis AI"),  # Output berupa Bar Chart
    title="🔍 Deteksi Hoax Naive Bayes",
    description="Sistem ini menggunakan Machine Learning untuk membedakan berita FAKTA dan HOAX. Masukkan teks berita untuk melihat seberapa yakin model terhadap prediksinya.",
    examples=contoh_berita,  # Fitur klik contoh
    theme="default")

# 4. JALANKAN APLIKASI
if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)
