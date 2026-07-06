import pandas as pd

# Load datasets
assessments = pd.read_csv("assessments.csv")
student_scores = pd.read_csv("studentAssessment.csv")

print("Files loaded successfully!\n")

# Merge datasets using assessment ID
merged = pd.merge(student_scores, assessments, on="id_assessment")

print("Merged dataset preview:\n")
print(merged.head())

# Check missing values
print("\nMissing values:\n")
print(merged.isnull().sum())

# Remove missing values
merged = merged.dropna()

# Save cleaned merged dataset
merged.to_csv("cleaned_merged_data.csv", index=False)

print("\nPreprocessing complete!")
print("Clean dataset saved as cleaned_merged_data.csv")