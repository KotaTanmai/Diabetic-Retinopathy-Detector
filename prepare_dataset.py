
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load original dataset
data = pd.read_csv("dataset/train.csv")

# Convert 5 classes into 2 classes
data["label"] = (data["diagnosis"] > 0).astype(int)

# Add image paths
data["image_path"] = data["id_code"].apply(
    lambda x: os.path.join("dataset", "train_images", x + ".png")
)

# Check whether images exist
missing = data[~data["image_path"].apply(os.path.exists)]

if len(missing) > 0:
    print("Missing images:", len(missing))
    print(missing["id_code"].head())
    raise SystemExit("Please check image filenames.")

# Split: 70% training, 30% temporary
train_data, temp_data = train_test_split(
    data,
    test_size=0.30,
    random_state=42,
    stratify=data["label"]
)

# Split temporary data into 15% validation and 15% testing
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=42,
    stratify=temp_data["label"]
)

# Save the three datasets
train_data.to_csv("dataset/train_split.csv", index=False)
val_data.to_csv("dataset/val_split.csv", index=False)
test_data.to_csv("dataset/test_split.csv", index=False)

print("Dataset preparation completed!")

print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Testing images:", len(test_data))

print("\nBinary class distribution:")
print(data["label"].value_counts().sort_index())

