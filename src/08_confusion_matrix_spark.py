import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# LOAD DATA (HASIL DARI 06)
df = pd.read_csv("output/prediction_logreg.csv")

print("Jumlah data evaluasi:", len(df))
print(df.head())

# MAP LABEL NUMERIK -> TEKS
label_map = {
    0.0: "positif",
    1.0: "netral",
    2.0: "negatif"
}

df["pred_label"] = df["prediction"].map(label_map)

# CONFUSION MATRIX
labels = ["positif", "netral", "negatif"]

cm = confusion_matrix(
    df["sentiment"],
    df["pred_label"],
    labels=labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix Sentimen Tokopedia")
plt.show()

# AKURASI
accuracy = accuracy_score(df["sentiment"], df["pred_label"])
print(f"Accuracy: {accuracy:.4f}")
