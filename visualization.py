import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("cleaned_merged_data.csv")

print("Dataset loaded for visualization\n")


plt.figure(figsize=(10,6))
plt.hist(data["score"], bins=40, edgecolor="black", alpha=0.7)

plt.title("Distribution of Student Scores")
plt.xlabel("Score")
plt.ylabel("Number of Students")

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
data["assessment_type"].value_counts().plot(kind="bar", color="skyblue")

plt.title("Number of Assessments by Type")
plt.xlabel("Assessment Type")
plt.ylabel("Count")

plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# -------------------------------
# 3️⃣ Score vs Assessment Weight
# -------------------------------
plt.figure(figsize=(10,6))
plt.scatter(data["weight"], data["score"], alpha=0.5)

plt.title("Score vs Assessment Weight")
plt.xlabel("Assessment Weight")
plt.ylabel("Student Score")

plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
data["weak_student"].value_counts().plot(kind="bar", color=["green","red"])

plt.title("Weak vs Strong Students")
plt.xlabel("Student Category")
plt.ylabel("Count")

plt.xticks([0,1], ["Strong","Weak"], rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

