import pandas as pd

# Load Data
df = pd.read_csv("data/processed/ulasan_clean.csv")

# Pastikan clean_text tidak NaN
df["clean_text"] = df["clean_text"].fillna("").astype(str)

# Kamus Sentimen
positive_words = [
    "bagus", "baik", "mantap", "puas", "cepat", "murah", "sesuai", "rekomendasi", "kuat", "legit"
]

negative_words = [
    "jelek", "buruk", "lama", "kecewa", "rusak", "mahal", "tidak sesuai", "kw", "bau"
]

# Fungsi Labeling
def label_sentiment(text):
    pos_count = sum(word in text for word in positive_words)
    neg_count = sum(word in text for word in negative_words)

    if pos_count > neg_count:
        return "positif"
    elif neg_count > pos_count:
        return "negatif"
    else:
        return "netral"

# Terapkan Labeling
df["sentiment"] = df["clean_text"].apply(label_sentiment)

# Simpan Data
df.to_csv("data/processed/ulasan_labeled.csv", index=False)

print("Labeling sentimen selesai")
print(df["sentiment"].value_counts())
