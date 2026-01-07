import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import glob

# Load CSV hasil Spark
csv_files = glob.glob("output/prediction_logreg.csv")

if not csv_files:
    raise FileNotFoundError("File prediksi CSV tidak ditemukan")

df = pd.concat([pd.read_csv(f) for f in csv_files])

print("Jumlah data:", len(df))
print(df["sentiment"].value_counts())

# Fungsi Top Words
def plot_top_words(texts, sentiment, top_n=10):
    words = " ".join(texts).split()
    counter = Counter(words)
    common_words = counter.most_common(top_n)

    words, counts = zip(*common_words)

    plt.figure(figsize=(8, 4))
    plt.bar(words, counts)
    plt.title(f"Top {top_n} Kata - Sentimen {sentiment}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Proses per sentimen
for sentiment in ["positif", "netral", "negatif"]:
    texts = df[df["sentiment"] == sentiment]["clean_text"].dropna()
    print(f"\nTop kata sentimen: {sentiment}")
    plot_top_words(texts, sentiment)
