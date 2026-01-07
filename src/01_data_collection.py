import pandas as pd

# Load Dataset
DATA_PATH = "data/raw/ulasan_tokopedia.csv"

df = pd.read_csv(DATA_PATH)

# Cek Data Awal
print("Jumlah data :", df.shape)
print("\nKolom dataset:")
print(df.columns)

print("\nContoh data:")
print(df.head())

# Cek Missing Value
print("\nMissing value:")
print(df.isnull().sum())
