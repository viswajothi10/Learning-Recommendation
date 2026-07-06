# 🎓 AI Learning Recommendation System

An AI-powered system that predicts student performance using **XGBoost** machine learning and provides **personalized learning recommendations** with YouTube videos and topic resources.

---

## 🚀 Features

- 🔍 **Predict** student performance (Weak / Strong) based on assessment data
- 📚 **Personalized topic recommendations** with direct links to learning resources
- 🎬 **YouTube video recommendations** matched to student level
- 📊 **Dashboard** with live charts — score distribution, weak vs strong, module averages
- 🕓 **Prediction history** with confidence scores
- 5 learning tiers: Critical → Weak → Average → Good → Excellent

---

## 🧠 ML Model

- **Algorithm:** XGBoost Classifier
- **Accuracy:** 92.68%
- **Class balancing:** SMOTE
- **Boundary:** Score ≤ 40 = Weak Student, Score > 40 = Strong Student

### Features Used
| Feature | Description |
|---|---|
| weight | Assessment weight (%) |
| date | Assessment due day |
| assessment_type | 0=Exam, 1=CMA, 2=TMA |
| code_module | Encoded module (0–6) |
| submission_delay | Days late (negative = early) |
| student_avg_score | Student's historical average score |
| assessment_difficulty | Same as weight |

---

## 🗂️ Project Structure

```
Learning-Recommendation/
├── backend/
│   ├── app.py              # Flask API
│   └── requirements.txt
├── frontend/
│   ├── public/
│   └── src/
│       ├── App.js          # React UI
│       ├── App.css
│       ├── index.js
│       └── index.css
├── preprocess.py           # Data preprocessing
├── model_training.py       # Model training
├── visualization.py        # Data visualization
├── cleaned_merged_data.csv
├── student_performance_model.pkl
├── scaler.pkl
├── start.bat               # One-click launcher
└── README.md
```

---

## ⚙️ Setup & Run

### Prerequisites
- Python 3.8+
- Node.js 16+

### 1. Clone the repository
```bash
git clone https://github.com/viswajothi10/Learning-Recommendation.git
cd Learning-Recommendation
```

### 2. Run preprocessing & train model
```bash
python preprocess.py
python model_training.py
```

### 3. Start Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 4. Start Frontend
```bash
cd frontend
npm install
npm start
```

### Or use the one-click launcher (Windows)
```
Double-click start.bat
```

Open **http://localhost:3000** in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Predict student performance |
| GET | `/stats` | Dataset statistics |
| GET | `/history` | Prediction history |
| DELETE | `/history` | Clear history |
| GET | `/health` | Health check |

---

## 🎯 Recommendation Tiers

| Score Range | Category | Resources |
|---|---|---|
| 1 – 20 | 🔴 Critical | Basic math, study skills, computer basics |
| 21 – 40 | 🟠 Weak | Python basics, intro to coding, AI concepts |
| 41 – 60 | 🟡 Average | Coding fundamentals, DSA, SQL, HTML |
| 61 – 75 | 🔵 Good | ML with Python, Scikit-Learn, Kaggle |
| 76 – 100 | 🟢 Excellent | Deep Learning, LLMs, MLOps, Research |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost, Scikit-Learn, SMOTE |
| Backend | Flask, Flask-CORS |
| Frontend | React.js, Chart.js |
| Data | Pandas, NumPy |
| Dataset | OULAD (Open University Learning Analytics) |
