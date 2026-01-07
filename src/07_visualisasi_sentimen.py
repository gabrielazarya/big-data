from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt

# Spark Session
spark = SparkSession.builder \
    .appName("Visualisasi Sentimen Tokopedia") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load data
df = spark.read.option("header", True).csv(
    "data/processed/ulasan_labeled.csv"
)

# FILTER DATA BERMASALAH
df = df.filter(col("sentiment").isNotNull())

# Normalisasi (AMAN)
df = df.filter(col("sentiment").isin("positif", "negatif", "netral"))

# Agregasi Big Data
sentiment_count = df.groupBy("sentiment").count()

# Convert ke Pandas
pdf = sentiment_count.toPandas()

# Cek terakhir (DEBUG AMAN)
print(pdf)

# Visualisasi
plt.figure()
plt.bar(pdf["sentiment"], pdf["count"])
plt.xlabel("Sentimen")
plt.ylabel("Jumlah Ulasan")
plt.title("Distribusi Sentimen Ulasan Tokopedia")
plt.show()

spark.stop()
