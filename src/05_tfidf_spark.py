from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

# 1. SPARK SESSION
spark = SparkSession.builder \
    .appName("TFIDF Tokopedia") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. LOAD DATA
df = spark.read.csv(
    "data/processed/ulasan_labeled.csv",
    header=True,
    inferSchema=True
)

print("Jumlah data awal:", df.count())

# 3. CLEAN NULL TEXT (WAJIB)
df = df.filter(col("clean_text").isNotNull())
df = df.filter(col("sentiment").isin("positif", "netral", "negatif"))

print("Jumlah data setelah cleaning:", df.count())

# 4. TOKENIZATION
tokenizer = Tokenizer(
    inputCol="clean_text",
    outputCol="words"
)
words_data = tokenizer.transform(df)

# 5. TF
hashingTF = HashingTF(
    inputCol="words",
    outputCol="rawFeatures",
    numFeatures=5000
)
tf_data = hashingTF.transform(words_data)

# 6. IDF
idf = IDF(
    inputCol="rawFeatures",
    outputCol="features"
)
idf_model = idf.fit(tf_data)
tfidf_data = idf_model.transform(tf_data)

# 7. OUTPUT RAPI (BUAT LAPORAN)
output_df = tfidf_data.select(
    "clean_text",
    "sentiment",
    size(col("words")).alias("jumlah_kata"),
    col("features").alias("tfidf_vector")
)

output_df.show(10, truncate=True)

print("TF-IDF berhasil dibentuk")

spark.stop()
