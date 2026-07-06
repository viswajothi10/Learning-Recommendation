import pandas as pd

print("Loading dataset...\n")

data = pd.read_csv("cleaned_merged_data.csv")

print("Dataset loaded successfully!\n")

print("Dataset shape before cleaning:", data.shape)

print("\nNull values in each column:\n")
print(data.isnull().sum())

data = data.dropna()

print("\nDataset shape after removing null values:", data.shape)

print("\nNull values after cleaning:\n")
print(data.isnull().sum())

data.to_csv("cleaned_merged_data.csv", index=False)

print("\nNull values removed successfully!")
print("Clean dataset saved as cleaned_merged_data.csv")