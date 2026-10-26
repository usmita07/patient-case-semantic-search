from datasets import load_dataset
from itertools import islice
import pandas as pd

print("Connecting to the dataset...")

# Stream the dataset instead of downloading the complete file
dataset = load_dataset(
    "aisc-team-a1/Asclepius-Synthetic-Clinical-Notes",
    split="train",
    streaming=True
)

print("Reading the first 2,000 records...")

# Read only the first 2,000 records
records = list(islice(dataset, 2000))

# Convert the records into a DataFrame
df = pd.DataFrame(records)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Display a few records
print(df.head(3))

# Save the sample
df.to_csv("data/patient_cases_large.csv", index=False)

print("\nSaved to data/patient_cases_large.csv")