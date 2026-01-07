from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, trim, when, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Spark Session (PALING RINGAN)
spark = SparkSession.builder \
    .appName("Sentimen Tokopedia") \
    .master("local[1]") \
    .config("spark.driver.memory", "512m") \
    .config("spark.executor.memory", "512m") \
    .config("spark.sql.shuffle.partitions", "1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. Load Data
df = spark.read.option("header", True) \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .csv("data/processed/ulasan_labeled.csv")

df = df.select("clean_text", "sentiment")

# Normalisasi label
df = df.withColumn("sentiment", lower(trim(col("sentiment"))))

df = df.withColumn(
    "sentiment",
    when(col("sentiment").contains("positif"), "positif")
    .when(col("sentiment").contains("negatif"), "negatif")
    .otherwise("netral")
)

df.groupBy("sentiment").count().show()

# Clean text
df = df.filter(col("clean_text").isNotNull())
df = df.filter(trim(col("clean_text")) != "")

print("Jumlah data setelah clean:", df.count())

# 3. Split Data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 4. Pipeline
label_indexer = StringIndexer(
    inputCol="sentiment",
    outputCol="label",
    handleInvalid="keep"
)

tokenizer = Tokenizer(
    inputCol="clean_text",
    outputCol="words"
)

hashingTF = HashingTF(
    inputCol="words",
    outputCol="features",
    numFeatures=1000
)

logreg = LogisticRegression(
    maxIter=20,
    regParam=0.1
)

pipeline = Pipeline(stages=[
    label_indexer,
    tokenizer,
    hashingTF,
    logreg
])

# 5. Training
model = pipeline.fit(train_data)

# 6. Prediction
predictions = model.transform(test_data)

predictions.select(
    "clean_text",
    "sentiment",
    "prediction"
).show(10, truncate=True)

# 7. Evaluation (BARU DI SINI)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

# Ambil sebagian kecil ke Pandas (aman, karena cuma hasil test)
pdf = predictions.select(
    "clean_text",
    "sentiment",
    "prediction"
).toPandas()

pdf.to_csv(
    "output/prediction_logreg.csv",
    index=False
)

print("Output berhasil disimpan via Pandas")


# 9. Stop Spark
spark.stop()
