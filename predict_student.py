import joblib
import numpy as np

model = joblib.load("student_performance_model.pkl")

print("Model loaded successfully\n")

student_data = np.array([[10, 20, 1, 3, 2, 65, 10]])

prediction = model.predict(student_data)

if prediction[0] == 1:
    print("Prediction: Weak Student")
    print("Recommendation: Provide extra practice materials and revision sessions.")
else:
    print("Prediction: Strong Student")
    print("Recommendation: Provide advanced learning resources.")