import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib

# -------------------------------
# Load dataset
# -------------------------------
print("Loading dataset...\n")

data = pd.read_csv("cleaned_merged_data.csv")

print("Dataset loaded successfully")
print("Dataset shape:", data.shape)

# -------------------------------
# Feature Engineering
# -------------------------------

data["submission_delay"] = data["date_submitted"] - data["date"]

data["student_avg_score"] = data.groupby("id_student")["score"].transform("mean")

data["assessment_difficulty"] = data["weight"]

# -------------------------------
# Encode categorical columns
# -------------------------------

le = LabelEncoder()

data["assessment_type"] = le.fit_transform(data["assessment_type"])
data["code_module"] = le.fit_transform(data["code_module"])

# -------------------------------
# Features and Target
# -------------------------------

X = data[[
    "weight",
    "date",
    "assessment_type",
    "code_module",
    "submission_delay",
    "student_avg_score",
    "assessment_difficulty"
]]

y = data["weak_student"]

# -------------------------------
# Feature Scaling
# -------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(X)

# -------------------------------
# Handle Class Imbalance
# -------------------------------

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nBalanced dataset:")
print(pd.Series(y_resampled).value_counts())

# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)

# -------------------------------
# Train XGBoost Model
# -------------------------------

model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training complete")

# -------------------------------
# Prediction
# -------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test, predictions))

# -------------------------------
# Feature Importance
# -------------------------------

importance = model.feature_importances_

plt.figure(figsize=(8,5))

plt.bar(range(len(importance)), importance)

plt.title("Feature Importance")

plt.xlabel("Feature Index")

plt.ylabel("Importance")

plt.show()

# -------------------------------
# Save Model and Scaler
# -------------------------------

joblib.dump(model, "student_performance_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully")