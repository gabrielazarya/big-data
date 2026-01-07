from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace, col

spark = SparkSession.builder \
    .appName("SentimentTokopedia") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "data/raw/ulasan_tokopedia.csv",
    header=True,
    inferSchema=True
)

df.printSchema()

# Case Folding
df_clean = df.withColumn(
    "clean_text",
    lower(col("komentar"))  
)

# Cleaning
df_clean = df_clean.withColumn(
    "clean_text",
    regexp_replace(col("clean_text"), r"http\S+|www\S+", "")
)

df_clean = df_clean.withColumn(
    "clean_text",
    regexp_replace(col("clean_text"), r"\d+", "")
)

df_clean = df_clean.withColumn(
    "clean_text",
    regexp_replace(col("clean_text"), r"[^a-zA-Z\s]", "")
)

# Simpan Hasil
df_clean.write.mode("overwrite").csv(
    "data/processed/ulasan_clean_spark",
    header=True
)

spark.stop()
print("Preprocessing Spark (tanpa stopword & stemming) selesai")
