import pandas as pd


print("Loading datasets...\n")

assessments = pd.read_csv("assessments.csv")
student_scores = pd.read_csv("studentAssessment.csv")

print("Datasets loaded successfully!\n")

print("Assessments dataset shape:", assessments.shape)
print("Student assessment dataset shape:", student_scores.shape)


merged = pd.merge(student_scores, assessments, on="id_assessment")

print("\nMerged dataset preview:\n")
print(merged.head())


print("\nMerged dataset shape:", merged.shape)
print("Number of rows:", merged.shape[0])
print("Number of columns:", merged.shape[1])


print("\nMissing values in merged dataset:\n")
print(merged.isnull().sum())


merged = merged.dropna()


merged["weak_student"] = merged["score"] <= 40  # score 0-40 = weak, score > 40 = strong

print("\nWeak student records:\n")
print(merged[merged["weak_student"] == True].head())
merged.to_csv("cleaned_merged_data.csv", index=False)

weak_students = merged[merged["weak_student"] == True]
weak_students.to_csv("weak_students.csv", index=False)

print("\nPreprocessing complete!")
print("Clean dataset saved as: cleaned_merged_data.csv")
print("Weak students saved as: weak_students.csv")