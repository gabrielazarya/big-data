import pandas as pd
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load Data
DATA_PATH = "data/raw/ulasan_tokopedia.csv"
df = pd.read_csv(DATA_PATH)

TEXT_COLUMN = "komentar" 

# Inisialisasi Stemmer
stemmer = StemmerFactory().create_stemmer()

# Kamus Normalisasi (contoh)
normalisasi_dict = {
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "bgt": "banget",
    "brg": "barang",
    "yg": "yang",
    "tdk": "tidak"
}

# Fungsi Preprocessing
def case_folding(text):
    return str(text).lower()

def cleaning(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text

def normalisasi(text):
    tokens = text.split()
    tokens = [normalisasi_dict.get(word, word) for word in tokens]
    return " ".join(tokens)

def stemming(text):
    return stemmer.stem(text)

# Pipeline Preprocessing
df["clean_text"] = df[TEXT_COLUMN].apply(case_folding)
df["clean_text"] = df["clean_text"].apply(cleaning)
df["clean_text"] = df["clean_text"].apply(normalisasi)
df["clean_text"] = df["clean_text"].apply(stemming)

# Simpan Data
OUTPUT_PATH = "data/processed/ulasan_clean.csv"
df.to_csv(OUTPUT_PATH, index=False)

print("Preprocessing TANPA stopword selesai!")
print("Data tersimpan di:", OUTPUT_PATH)
