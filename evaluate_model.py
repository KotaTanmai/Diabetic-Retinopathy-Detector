
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Load test dataset
test_df = pd.read_csv("dataset/test_split.csv")

# Load trained CNN model
model = load_model("model/retinopathy_model.keras")

actual_classes = []
predicted_classes = []

# Predict test images
for _, row in test_df.iterrows():

    image_path = row["image_path"]

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0][0]

    actual_classes.append(int(row["label"]))
    predicted_classes.append(1 if prediction >= 0.5 else 0)

# Calculate performance
accuracy = accuracy_score(actual_classes, predicted_classes)
precision = precision_score(actual_classes, predicted_classes)
recall = recall_score(actual_classes, predicted_classes)
f1 = f1_score(actual_classes, predicted_classes)

print("\nMODEL PERFORMANCE")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    actual_classes,
    predicted_classes,
    target_names=["No DR", "DR Detected"]
))

os.makedirs("model", exist_ok=True)

# 1. Confusion Matrix
cm = confusion_matrix(actual_classes, predicted_classes)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["No DR", "DR"])
plt.yticks([0, 1], ["No DR", "DR"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.close()

# 2. Performance Metrics Graph
metrics = {
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1 Score": f1 * 100
}

plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values())
plt.title("CNN Model Performance")
plt.ylabel("Score (%)")
plt.ylim(0, 100)

for i, value in enumerate(metrics.values()):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.savefig("model/performance_metrics.png")
plt.close()

print("\nBoth graphs saved successfully!")
