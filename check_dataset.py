
import os
import pandas as pd

dataset_path = "dataset"

train_csv = os.path.join(dataset_path, "train.csv")
train_images = os.path.join(dataset_path, "train_images")

print("Checking dataset...")

if os.path.exists(train_csv):
    data = pd.read_csv(train_csv)

    print("\nDataset loaded successfully!")
    print("Total training records:", len(data))
    print("\nColumn names:", list(data.columns))
    print("\nFirst 5 records:")
    print(data.head())

    print("\nClass distribution:")
    print(data["diagnosis"].value_counts().sort_index())

else:
    print("train.csv not found! Check your dataset folder.")

if os.path.exists(train_images):
    images = os.listdir(train_images)
    print("\nTotal training images:", len(images))
else:
    print("train_images folder not found!")
