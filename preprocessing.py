import pandas as pd
import re
import string

# LOAD DATA
df = pd.read_excel('datasetberita.xlsx')
print(df['hoax'].value_counts())
print("\nJumlah data")
print(df.shape)

# HAPUS KOLOM YANG TIDAK DIGUNAKAN (Attribute Selection)
print("\n=== DATAFRAME SEBELUM KOLUM DIHAPUS ===")
print("\nKolom yang tersedia:")
print(df.columns)

df = df.drop(columns=['url', 'judul', 'tanggal', 'isi_berita', 'Narasi'])

print("\n=== DATAFRAME SETELAH KOLUM DIHAPUS ===")
print("\nKolom yang tersisa:")
print(df.columns)

# CEK MISSING VALUE
print("\n=== CEK MISSING VALUE ===")
print(df.isnull().sum())

# Hapus baris yang missing
df = df.dropna(subset=['Clean Narasi', 'hoax'])
print("\nJumlah data setelah drop missing:")
print(df.shape)

# CEK DISTRIBUSI LABEL
print("\n=== DISTRIBUSI LABEL SEBELUM BALANCING ===")
print(df['hoax'].value_counts())

# BALANCING DATA
df_0 = df[df['hoax'] == 0]   # Fakta
df_1 = df[df['hoax'] == 1]   # Hoax

min_count = min(len(df_0), len(df_1))
df_seimbang = pd.concat([
    df_0.sample(n=min_count, random_state=42),
    df_1.sample(n=min_count, random_state=42)
])

print("\n=== DISTRIBUSI LABEL SETELAH BALANCING ===")
print(df_seimbang['hoax'].value_counts())

# STOPWORDS INDONESIA
stopwords_indonesia = [
    "yang","di","dan","itu","ini","dari","ke","akan","pada","juga","adalah",
    "karena","untuk","dengan","saya","kamu","kita","mereka","ada","seperti",
    "tapi","kalau","atau","tidak","bisa","sudah","lagi","mau","sama","tak",
    "bukan","belum","hanya","para","namun","oleh","telah","bagi","ia","dia",
    "saat","dalam","banyak","setelah","kepada","sebagai","bahwa","sebuah",
    "tentang","maka","serta","pun","apakah","mengapa","siapa","dimana",
    "kapan","bagaimana","jika","sehingga","secara","hal","tersebut",
    "kami","anda","menjadi"
]

# Pembersihan Teks
def bersihkan_teks(teks):
    if not isinstance(teks, str):
        return ""

    teks = teks.lower() #Case Folding
    teks = re.sub(r'\d+', '', teks) #Hapus Angka
    teks = teks.translate(str.maketrans('', '', string.punctuation)) #Hapus Simbol
    teks = teks.strip() #Menghapus spasi berlebih di awal/akhir

    # hapus stopwords
    words = [w for w in teks.split() if w not in stopwords_indonesia]

    # hapus kata terlalu pendek (< 3 huruf)
    words = [w for w in words if len(w) >= 3]

    return " ".join(words)

# Terapkan cleaning
df_seimbang['Clean Narasi'] = df_seimbang['Clean Narasi'].apply(bersihkan_teks)

# CEK HASIL AKHIR
print("\n=== CEK MISSING VALUE SETELAH PROSES ===")
print(df_seimbang.isnull().sum())

print("\nContoh hasil pembersihan:")
print(df_seimbang[['Clean Narasi', 'hoax']].head())

# SIMPAN HASIL
df_seimbang.to_csv('datasetberita.csv', index=False)
print("\nDataset final berhasil disimpan -> datasetberita.csv")