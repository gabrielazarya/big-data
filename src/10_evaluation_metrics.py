import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load CSV hasil prediksi Spark
df = pd.read_csv("output/prediction_logreg.csv")

# Mapping label numerik ke sentimen
label_map = {
    0.0: "positif",
    1.0: "netral",
    2.0: "negatif"
}

df["prediction_label"] = df["prediction"].map(label_map)

# Ground truth
y_true = df["sentiment"]
y_pred = df["prediction_label"]

# Classification Report
print("=== EVALUATION REPORT ===")
print(classification_report(
    y_true,
    y_pred,
    labels=["positif", "netral", "negatif"],
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=["positif", "netral", "negatif"])
print("=== CONFUSION MATRIX ===")
print(cm)