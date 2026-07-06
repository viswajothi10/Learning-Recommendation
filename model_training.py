import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("cleaned_merged_data.csv")
print("Dataset loaded successfully")

# -------------------------------
# Feature Engineering
# -------------------------------

# submission delay
data["submission_delay"] = data["date_submitted"] - data["date"]

# student historical average score
data["student_avg_score"] = data.groupby("id_student")["score"].transform("mean")

# assessment difficulty
data["assessment_difficulty"] = data["weight"]

# -------------------------------
# Encode categorical variables
# -------------------------------
le = LabelEncoder()
data["assessment_type"] = le.fit_transform(data["assessment_type"])
data["code_module"] = le.fit_transform(data["code_module"])

# -------------------------------
# Features
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
# Feature scaling
# -------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# Handle class imbalance
# -------------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Balanced dataset:")
print(pd.Series(y_resampled).value_counts())

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# -------------------------------
# Train XGBoost model
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

print("Model training complete")

# -------------------------------
# Predictions
# -------------------------------
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))