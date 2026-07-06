import pandas as pd

# STEP 1: Load the dataset
print("Loading dataset...\n")

data = pd.read_csv("cleaned_merged_data.csv")

print("Dataset loaded successfully!\n")

# STEP 2: Show dataset shape before cleaning
print("Dataset shape before cleaning:", data.shape)

# STEP 3: Check null values
print("\nNull values in each column:\n")
print(data.isnull().sum())

# STEP 4: Drop rows with null values
data = data.dropna()

# STEP 5: Check dataset shape after cleaning
print("\nDataset shape after removing null values:", data.shape)

# STEP 6: Verify no null values remain
print("\nNull values after cleaning:\n")
print(data.isnull().sum())

# STEP 7: Save cleaned dataset
data.to_csv("cleaned_merged_data.csv", index=False)

print("\nNull values removed successfully!")
print("Clean dataset saved as cleaned_merged_data.csv")