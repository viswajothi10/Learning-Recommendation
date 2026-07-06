from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "student_performance_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

history = []

# ─── Dynamic Recommendation Engine ───────────────────────────────────────────

WEAK_LEVELS = {
    "critical": {  # avg_score 1–20 → absolute basics
        "tips": [
            "Attend every class — missing even one session sets you back",
            "Start from scratch: numbers, basic operations, simple sentences",
            "Spend 1 hour daily watching beginner YouTube tutorials",
            "Use flashcards to memorize key definitions and formulas",
            "Ask your teacher for a personal recovery study plan",
            "Revise the same topic 3 times before moving to the next",
            "Avoid distractions — put your phone away during study time"
        ],
        "topics": [
            { "title": "Number Systems & Basic Arithmetic",         "icon": "🔢", "level": "Beginner",     "url": "https://www.khanacademy.org/math/arithmetic" },
            { "title": "Introduction to Algebra",                   "icon": "📐", "level": "Beginner",     "url": "https://www.khanacademy.org/math/algebra" },
            { "title": "Basic Statistics: Mean, Median, Mode",      "icon": "📊", "level": "Beginner",     "url": "https://www.khanacademy.org/math/statistics-probability" },
            { "title": "Reading & Writing Fundamentals",            "icon": "📖", "level": "Beginner",     "url": "https://www.khanacademy.org/ela" },
            { "title": "Study Skills & Active Recall Techniques",   "icon": "🧠", "level": "Beginner",     "url": "https://www.coursera.org/learn/learning-how-to-learn" },
            { "title": "Basic Computer Skills & MS Office",         "icon": "💻", "level": "Beginner",     "url": "https://edu.gcfglobal.org/en/topics/computers/" },
            { "title": "Introduction to Internet & Digital Tools",  "icon": "🌐", "level": "Beginner",     "url": "https://edu.gcfglobal.org/en/internetbasics/" },
            { "title": "Simple Problem Solving Strategies",         "icon": "🧩", "level": "Beginner",     "url": "https://www.khanacademy.org/computing/computer-science" },
        ],
        "videos": [
            { "title": "How to Study Effectively – Beginner Guide",          "channel": "Thomas Frank",        "url": "https://www.youtube.com/watch?v=IlU-zDU6aQ0", "thumb": "https://img.youtube.com/vi/IlU-zDU6aQ0/mqdefault.jpg" },
            { "title": "Basic Math Skills You Need to Know",                 "channel": "Math Antics",         "url": "https://www.youtube.com/watch?v=VgDe_D8ojxw", "thumb": "https://img.youtube.com/vi/VgDe_D8ojxw/mqdefault.jpg" },
            { "title": "Introduction to Algebra for Beginners",              "channel": "Khan Academy",        "url": "https://www.youtube.com/watch?v=NybHckSEQBI", "thumb": "https://img.youtube.com/vi/NybHckSEQBI/mqdefault.jpg" },
            { "title": "How to Take Notes Effectively",                      "channel": "Mike and Matty",      "url": "https://www.youtube.com/watch?v=AjoxkxM_I5g", "thumb": "https://img.youtube.com/vi/AjoxkxM_I5g/mqdefault.jpg" },
            { "title": "Active Recall & Spaced Repetition Study Method",     "channel": "Ali Abdaal",          "url": "https://www.youtube.com/watch?v=ukLnPbIffxE", "thumb": "https://img.youtube.com/vi/ukLnPbIffxE/mqdefault.jpg" },
            { "title": "Computer Basics Full Course for Beginners",          "channel": "GCFLearnFree",        "url": "https://www.youtube.com/watch?v=y2kg3MOk1sY", "thumb": "https://img.youtube.com/vi/y2kg3MOk1sY/mqdefault.jpg" },
        ]
    },
    "weak": {  # avg_score 21–41 → foundational concepts + intro to coding
        "tips": [
            "Review all previous assessments and list your weak topics",
            "Practice 5 questions daily from each weak subject area",
            "Start learning Python basics — it's the easiest first language",
            "Use Khan Academy or W3Schools for free structured learning",
            "Join a study group or find an online peer learning community",
            "Break study sessions into 25-minute Pomodoro blocks",
            "Build a simple project: calculator or quiz app to apply learning"
        ],
        "topics": [
            { "title": "Python Basics: Variables, Input & Output",    "icon": "🐍", "level": "Intro Coding", "url": "https://www.w3schools.com/python/python_variables.asp" },
            { "title": "Conditional Statements (if/else)",             "icon": "🔀", "level": "Intro Coding", "url": "https://www.w3schools.com/python/python_conditions.asp" },
            { "title": "Loops: for & while",                          "icon": "🔁", "level": "Intro Coding", "url": "https://www.w3schools.com/python/python_for_loops.asp" },
            { "title": "Functions & Basic Modules",                   "icon": "⚙️", "level": "Intro Coding", "url": "https://www.w3schools.com/python/python_functions.asp" },
            { "title": "Lists, Tuples & Dictionaries",                "icon": "📋", "level": "Intro Coding", "url": "https://www.w3schools.com/python/python_lists.asp" },
            { "title": "Basic Mathematics for Computing",             "icon": "📐", "level": "Maths",        "url": "https://www.khanacademy.org/math/cc-sixth-grade-math" },
            { "title": "Introduction to Algorithms & Flowcharts",     "icon": "🗺️", "level": "Concepts",    "url": "https://www.geeksforgeeks.org/introduction-to-algorithms/" },
            { "title": "What is AI? — Concepts & Real-world Uses",    "icon": "🤖", "level": "Concepts",    "url": "https://www.ibm.com/topics/artificial-intelligence" },
            { "title": "What is Data Science? — Overview",            "icon": "📊", "level": "Concepts",    "url": "https://www.ibm.com/topics/data-science" },
            { "title": "Internet Basics & How Websites Work",         "icon": "🌐", "level": "Concepts",    "url": "https://edu.gcfglobal.org/en/internetbasics/" },
            { "title": "Scratch / Block Coding for Logic Building",   "icon": "🧱", "level": "Intro Coding", "url": "https://scratch.mit.edu/" },
            { "title": "Problem Solving with Pseudocode",             "icon": "🧩", "level": "Concepts",    "url": "https://www.geeksforgeeks.org/how-to-write-a-pseudo-code/" },
        ],
        "videos": [
            { "title": "Python for Beginners – Full Course",                  "channel": "Programming with Mosh", "url": "https://www.youtube.com/watch?v=kqtD5dpn9C8", "thumb": "https://img.youtube.com/vi/kqtD5dpn9C8/mqdefault.jpg" },
            { "title": "Python Basics in 1 Hour",                            "channel": "Corey Schafer",         "url": "https://www.youtube.com/watch?v=YYXdXT2l-Gg", "thumb": "https://img.youtube.com/vi/YYXdXT2l-Gg/mqdefault.jpg" },
            { "title": "What is Artificial Intelligence? – Explained Simply", "channel": "CrashCourse",          "url": "https://www.youtube.com/watch?v=a0_lo_GDcFw", "thumb": "https://img.youtube.com/vi/a0_lo_GDcFw/mqdefault.jpg" },
            { "title": "Introduction to Data Science for Beginners",         "channel": "freeCodeCamp",         "url": "https://www.youtube.com/watch?v=ua-CiDNNj30", "thumb": "https://img.youtube.com/vi/ua-CiDNNj30/mqdefault.jpg" },
            { "title": "Algorithms & Flowcharts – Beginner Explained",       "channel": "CS Dojo",              "url": "https://www.youtube.com/watch?v=rL8X2mlNHPM", "thumb": "https://img.youtube.com/vi/rL8X2mlNHPM/mqdefault.jpg" },
            { "title": "The Pomodoro Technique – Study Smarter",             "channel": "Thomas Frank",         "url": "https://www.youtube.com/watch?v=mNBmG24djoY", "thumb": "https://img.youtube.com/vi/mNBmG24djoY/mqdefault.jpg" },
            { "title": "Khan Academy – Intro to Programming",                "channel": "Khan Academy",         "url": "https://www.youtube.com/watch?v=FCMxA3m_Imc", "thumb": "https://img.youtube.com/vi/FCMxA3m_Imc/mqdefault.jpg" },
            { "title": "How the Internet Works – Explained for Beginners",   "channel": "Lesics",               "url": "https://www.youtube.com/watch?v=x3c1ih2NJEg", "thumb": "https://img.youtube.com/vi/x3c1ih2NJEg/mqdefault.jpg" },
        ]
    },
    "late": {  # submission_delay > 5
        "tips": [
            "Create a weekly assignment tracker",
            "Set phone reminders 3 days before deadlines",
            "Break large tasks into daily mini-goals",
            "Avoid last-minute submissions — start early",
            "Use a planner or digital calendar consistently"
        ],
        "topics": [
            { "title": "Time Management & Productivity",        "icon": "⏰", "level": "Life Skill", "url": "https://todoist.com/productivity-methods" },
            { "title": "Goal Setting & Planning Techniques",    "icon": "🎯", "level": "Life Skill", "url": "https://www.mindtools.com/pages/article/newHTE_90.htm" },
            { "title": "Overcoming Procrastination",            "icon": "🚀", "level": "Life Skill", "url": "https://www.verywellmind.com/the-psychology-of-procrastination-2795944" },
            { "title": "Assignment Planning Strategies",        "icon": "📋", "level": "Life Skill", "url": "https://www.oxfordlearning.com/how-to-plan-assignments/" },
        ],
        "videos": [
            { "title": "How to Stop Procrastinating – 5 Steps",         "channel": "Thomas Frank",       "url": "https://www.youtube.com/watch?v=4aYVLpY5LkA", "thumb": "https://img.youtube.com/vi/4aYVLpY5LkA/mqdefault.jpg" },
            { "title": "Time Management for Students",                  "channel": "Mike and Matty",     "url": "https://www.youtube.com/watch?v=iDbdXTMnOmE", "thumb": "https://img.youtube.com/vi/iDbdXTMnOmE/mqdefault.jpg" },
            { "title": "How to Plan Your Week Effectively",             "channel": "Ali Abdaal",         "url": "https://www.youtube.com/watch?v=W9k0OhJkjQ0", "thumb": "https://img.youtube.com/vi/W9k0OhJkjQ0/mqdefault.jpg" },
        ]
    }
}

STRONG_LEVELS = {
    "average": {  # avg_score 42–60 → coding basics + fundamentals of all concepts
        "tips": [
            "Start coding daily — even 30 minutes builds strong habits",
            "Learn Python basics first: variables, loops, functions, OOP",
            "Understand core CS concepts: arrays, sorting, recursion",
            "Practice coding problems on HackerRank or LeetCode (Easy level)",
            "Build small projects: calculator, to-do app, number guessing game",
            "Learn HTML/CSS basics to understand how web works",
            "Study database basics: SQL queries, tables, joins"
        ],
        "topics": [
            { "title": "Python Programming Basics",                  "icon": "🐍", "level": "Coding", "url": "https://www.w3schools.com/python/" },
            { "title": "Variables, Loops & Functions",               "icon": "🔁", "level": "Coding", "url": "https://www.learnpython.org/" },
            { "title": "Object-Oriented Programming (OOP)",          "icon": "📦", "level": "Coding", "url": "https://realpython.com/python3-object-oriented-programming/" },
            { "title": "Data Structures: Arrays, Lists, Stacks",     "icon": "🗂️", "level": "Coding", "url": "https://www.geeksforgeeks.org/data-structures/" },
            { "title": "Algorithms: Sorting & Searching",            "icon": "🔍", "level": "Coding", "url": "https://www.geeksforgeeks.org/sorting-algorithms/" },
            { "title": "SQL & Database Fundamentals",                "icon": "🗄️", "level": "Coding", "url": "https://www.w3schools.com/sql/" },
            { "title": "HTML & CSS Basics",                          "icon": "🌐", "level": "Web",    "url": "https://www.w3schools.com/html/" },
            { "title": "Git & GitHub for Beginners",                 "icon": "🐙", "level": "Tools",  "url": "https://docs.github.com/en/get-started" },
            { "title": "Introduction to AI & Machine Learning",      "icon": "🤖", "level": "AI/ML", "url": "https://www.coursera.org/learn/machine-learning" },
            { "title": "Mathematics for Programming (Logic & Sets)", "icon": "📐", "level": "Maths",  "url": "https://www.khanacademy.org/math/discrete-math" },
            { "title": "Problem Solving & Pseudocode",               "icon": "🧩", "level": "Coding", "url": "https://www.hackerrank.com/domains/tutorials/10-days-of-javascript" },
            { "title": "Introduction to Data Science",               "icon": "📊", "level": "Data",   "url": "https://www.kaggle.com/learn/intro-to-programming" },
        ],
        "videos": [
            { "title": "Python Full Course for Beginners",                   "channel": "Programming with Mosh", "url": "https://www.youtube.com/watch?v=_uQrJ0TkZlc", "thumb": "https://img.youtube.com/vi/_uQrJ0TkZlc/mqdefault.jpg" },
            { "title": "Data Structures & Algorithms for Beginners",         "channel": "freeCodeCamp",          "url": "https://www.youtube.com/watch?v=8hly31xKli0", "thumb": "https://img.youtube.com/vi/8hly31xKli0/mqdefault.jpg" },
            { "title": "Object Oriented Programming – Full Course",          "channel": "freeCodeCamp",          "url": "https://www.youtube.com/watch?v=Ej_02ICOIgs", "thumb": "https://img.youtube.com/vi/Ej_02ICOIgs/mqdefault.jpg" },
            { "title": "SQL Full Course for Beginners",                      "channel": "Programming with Mosh", "url": "https://www.youtube.com/watch?v=7S_tz1z_5bA", "thumb": "https://img.youtube.com/vi/7S_tz1z_5bA/mqdefault.jpg" },
            { "title": "HTML & CSS Full Course – Beginner to Pro",           "channel": "SuperSimpleDev",        "url": "https://www.youtube.com/watch?v=G3e-cpL7ofc", "thumb": "https://img.youtube.com/vi/G3e-cpL7ofc/mqdefault.jpg" },
            { "title": "Git & GitHub Crash Course",                          "channel": "Traversy Media",        "url": "https://www.youtube.com/watch?v=RGOj5yH7evk", "thumb": "https://img.youtube.com/vi/RGOj5yH7evk/mqdefault.jpg" },
            { "title": "Introduction to Machine Learning for Beginners",     "channel": "Google Developers",     "url": "https://www.youtube.com/watch?v=gmvvaobm7eQ", "thumb": "https://img.youtube.com/vi/gmvvaobm7eQ/mqdefault.jpg" },
            { "title": "CS50 – Introduction to Computer Science (Harvard)",  "channel": "CS50",                  "url": "https://www.youtube.com/watch?v=8mAITcNt710", "thumb": "https://img.youtube.com/vi/8mAITcNt710/mqdefault.jpg" },
        ]
    },
    "good": {  # avg_score 60–75 → AI/ML beginner path
        "tips": [
            "Start the Python for ML roadmap — numpy, pandas, scikit-learn",
            "Complete Andrew Ng's Machine Learning course on Coursera",
            "Build your first ML project: house price prediction or spam classifier",
            "Practice on Kaggle beginner competitions to apply your skills",
            "Learn Git & GitHub to version-control your AI projects",
            "Study linear algebra and statistics — the math behind ML"
        ],
        "topics": [
            { "title": "Python for Machine Learning",                        "icon": "🐍", "level": "AI/ML", "url": "https://www.w3schools.com/python/" },
            { "title": "Supervised Learning (Regression & Classification)",  "icon": "🤖", "level": "AI/ML", "url": "https://scikit-learn.org/stable/supervised_learning.html" },
            { "title": "Data Preprocessing & Feature Engineering",           "icon": "⚙️", "level": "AI/ML", "url": "https://www.kaggle.com/learn/data-cleaning" },
            { "title": "Scikit-Learn & Model Evaluation",                    "icon": "📊", "level": "AI/ML", "url": "https://scikit-learn.org/stable/getting_started.html" },
            { "title": "Exploratory Data Analysis (EDA)",                    "icon": "🔍", "level": "AI/ML", "url": "https://www.kaggle.com/learn/pandas" },
            { "title": "Linear & Logistic Regression",                       "icon": "📈", "level": "AI/ML", "url": "https://www.geeksforgeeks.org/ml-linear-regression/" },
            { "title": "Decision Trees & Random Forests",                    "icon": "🌳", "level": "AI/ML", "url": "https://www.geeksforgeeks.org/decision-tree/" },
            { "title": "Kaggle Competitions for Beginners",                  "icon": "🏆", "level": "AI/ML", "url": "https://www.kaggle.com/competitions" },
        ],
        "videos": [
            { "title": "Machine Learning Full Course for Beginners",         "channel": "freeCodeCamp",      "url": "https://www.youtube.com/watch?v=NWONeJKn6kc", "thumb": "https://img.youtube.com/vi/NWONeJKn6kc/mqdefault.jpg" },
            { "title": "Python for Data Science and Machine Learning",       "channel": "Simplilearn",       "url": "https://www.youtube.com/watch?v=7eh4d6sabA0", "thumb": "https://img.youtube.com/vi/7eh4d6sabA0/mqdefault.jpg" },
            { "title": "Scikit-Learn Crash Course – ML with Python",         "channel": "Traversy Media",    "url": "https://www.youtube.com/watch?v=0B5eIE_1vpU", "thumb": "https://img.youtube.com/vi/0B5eIE_1vpU/mqdefault.jpg" },
            { "title": "Statistics for Machine Learning (Full Course)",      "channel": "freeCodeCamp",      "url": "https://www.youtube.com/watch?v=xxpc-HPKN28", "thumb": "https://img.youtube.com/vi/xxpc-HPKN28/mqdefault.jpg" },
            { "title": "Pandas & NumPy Full Tutorial",                       "channel": "Keith Galli",       "url": "https://www.youtube.com/watch?v=vmEHCJofslg", "thumb": "https://img.youtube.com/vi/vmEHCJofslg/mqdefault.jpg" },
            { "title": "Kaggle Intro to Machine Learning",                   "channel": "Kaggle",            "url": "https://www.youtube.com/watch?v=i_LwzRVP7bg", "thumb": "https://img.youtube.com/vi/i_LwzRVP7bg/mqdefault.jpg" },
        ]
    },
    "excellent": {  # avg_score > 75 → Advanced AI/ML/DL path
        "tips": [
            "Dive into Deep Learning — build CNNs, RNNs, and Transformers",
            "Complete the DeepLearning.AI specialization on Coursera",
            "Implement research papers from scratch using PyTorch or TensorFlow",
            "Contribute to open-source AI projects on GitHub (HuggingFace, scikit-learn)",
            "Explore Large Language Models (LLMs) — fine-tune GPT or BERT",
            "Publish your own AI project or research paper",
            "Apply for AI internships or competitions like Google AI Challenge"
        ],
        "topics": [
            { "title": "Deep Learning & Neural Networks",           "icon": "🧠", "level": "Deep Learning", "url": "https://www.deeplearning.ai/" },
            { "title": "Convolutional Neural Networks (CNN)",        "icon": "🖼️", "level": "Deep Learning", "url": "https://cs231n.github.io/" },
            { "title": "Recurrent Neural Networks & LSTM",           "icon": "🔄", "level": "Deep Learning", "url": "https://colah.github.io/posts/2015-08-Understanding-LSTMs/" },
            { "title": "Transformers & Attention Mechanism",         "icon": "⚡", "level": "Deep Learning", "url": "https://huggingface.co/learn/nlp-course/chapter1/1" },
            { "title": "Large Language Models (LLMs) & GPT",         "icon": "💬", "level": "Generative AI", "url": "https://platform.openai.com/docs/introduction" },
            { "title": "Natural Language Processing (NLP)",          "icon": "📝", "level": "AI/ML",        "url": "https://www.nltk.org/" },
            { "title": "Computer Vision with OpenCV & YOLO",         "icon": "👁️", "level": "AI/ML",        "url": "https://docs.opencv.org/4.x/d9/df8/tutorial_root.html" },
            { "title": "Reinforcement Learning",                     "icon": "🎮", "level": "Advanced AI",   "url": "https://spinningup.openai.com/en/latest/" },
            { "title": "MLOps & Model Deployment (FastAPI, Docker)", "icon": "🚀", "level": "MLOps",        "url": "https://mlflow.org/docs/latest/index.html" },
            { "title": "PyTorch & TensorFlow Advanced",              "icon": "🔥", "level": "Deep Learning", "url": "https://pytorch.org/tutorials/" },
            { "title": "Generative AI & Diffusion Models",           "icon": "🎨", "level": "Generative AI", "url": "https://huggingface.co/models" },
            { "title": "AI Research Paper Implementation",           "icon": "📄", "level": "Research",      "url": "https://paperswithcode.com/" },
        ],
        "videos": [
            { "title": "Deep Learning Specialization – Andrew Ng",           "channel": "DeepLearning.AI",   "url": "https://www.youtube.com/watch?v=CS4cs9xVecg", "thumb": "https://img.youtube.com/vi/CS4cs9xVecg/mqdefault.jpg" },
            { "title": "PyTorch Full Course – Deep Learning",                "channel": "freeCodeCamp",      "url": "https://www.youtube.com/watch?v=V_xro1bcAuA", "thumb": "https://img.youtube.com/vi/V_xro1bcAuA/mqdefault.jpg" },
            { "title": "Transformers & Attention – Illustrated Guide",       "channel": "Andrej Karpathy",   "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY", "thumb": "https://img.youtube.com/vi/kCc8FmEb1nY/mqdefault.jpg" },
            { "title": "Natural Language Processing with Python",            "channel": "freeCodeCamp",      "url": "https://www.youtube.com/watch?v=X2vAabgKiuM", "thumb": "https://img.youtube.com/vi/X2vAabgKiuM/mqdefault.jpg" },
            { "title": "Computer Vision Full Course – OpenCV & YOLO",        "channel": "Nicholas Renotte",  "url": "https://www.youtube.com/watch?v=N81PCpADwKQ", "thumb": "https://img.youtube.com/vi/N81PCpADwKQ/mqdefault.jpg" },
            { "title": "Generative AI Full Course – LLMs & ChatGPT",         "channel": "freeCodeCamp",      "url": "https://www.youtube.com/watch?v=mEsleV16qdo", "thumb": "https://img.youtube.com/vi/mEsleV16qdo/mqdefault.jpg" },
            { "title": "Reinforcement Learning Full Course",                 "channel": "freeCodeCamp",      "url": "https://www.youtube.com/watch?v=ELE2_Mftqoc", "thumb": "https://img.youtube.com/vi/ELE2_Mftqoc/mqdefault.jpg" },
            { "title": "MLOps Full Course – Deploy ML Models",               "channel": "Krish Naik",        "url": "https://www.youtube.com/watch?v=9BgIDqAzfuA", "thumb": "https://img.youtube.com/vi/9BgIDqAzfuA/mqdefault.jpg" },
        ]
    }
}


def get_dynamic_recommendations(prediction, avg_score, submission_delay):
    avg_score = float(avg_score)
    submission_delay = float(submission_delay)

    if prediction == 1:  # Weak student — 2 tiers
        if avg_score <= 20:
            level = WEAK_LEVELS["critical"]
            level_label = "Critical (Score 1–20) — Needs Immediate Support"
        else:
            level = WEAK_LEVELS["weak"]
            level_label = "Weak (Score 21–41) — Foundational Concepts & Intro to Coding"

        extra_topics = []
        extra_videos = []
        extra_tips   = []
        if submission_delay > 5:
            extra_topics = WEAK_LEVELS["late"]["topics"]
            extra_videos = WEAK_LEVELS["late"]["videos"]
            extra_tips   = WEAK_LEVELS["late"]["tips"][:2]

        return {
            "level_label": level_label,
            "tips":   level["tips"] + extra_tips,
            "topics": level["topics"] + extra_topics,
            "videos": level["videos"] + extra_videos,
        }
    else:  # Strong student — 3 tiers based on avg_score
        if avg_score > 75:
            level = STRONG_LEVELS["excellent"]
            level_label = "Excellent — Advanced AI / ML / Deep Learning Path"
        elif avg_score > 60:
            level = STRONG_LEVELS["good"]
            level_label = "Good — Start Your AI & Machine Learning Journey"
        else:
            level = STRONG_LEVELS["average"]
            level_label = "Average — Build Coding Basics & Core CS Concepts"

        return {
            "level_label": level_label,
            "tips":   level["tips"],
            "topics": level["topics"],
            "videos": level["videos"],
        }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()
        features = [
            float(body["weight"]),
            float(body["date"]),
            float(body["assessment_type"]),
            float(body["code_module"]),
            float(body["submission_delay"]),
            float(body["student_avg_score"]),
            float(body["assessment_difficulty"])
        ]
        data = np.array([features])
        data_scaled = scaler.transform(data)
        prob = model.predict_proba(data_scaled)[0]
        weak_prob = float(prob[1])

        # score <= 40 = weak, score > 40 = strong (matches dataset definition)
        prediction = 1 if weak_prob >= 0.5 else 0
        confidence = round((weak_prob if prediction == 1 else float(prob[0])) * 100, 2)

        label = "Weak Student" if prediction == 1 else "Strong Student"
        recs = get_dynamic_recommendations(prediction, body["student_avg_score"], body["submission_delay"])

        entry = {
            "id": len(history) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": body,
            "prediction": prediction,
            "label": label,
            "confidence": confidence
        }
        history.append(entry)

        return jsonify({
            "prediction": prediction,
            "label": label,
            "confidence": confidence,
            "level_label": recs["level_label"],
            "tips":   recs["tips"],
            "topics": recs["topics"],
            "videos": recs["videos"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(list(reversed(history)))


@app.route("/history", methods=["DELETE"])
def clear_history():
    history.clear()
    return jsonify({"status": "cleared"})


@app.route("/stats", methods=["GET"])
def stats():
    try:
        csv_path = os.path.join(BASE_DIR, "cleaned_merged_data.csv")
        df = pd.read_csv(csv_path)

        weak_count   = int(df["weak_student"].sum())
        strong_count = int(len(df) - weak_count)
        avg_score    = round(float(df["score"].mean()), 2)
        total        = len(df)

        score_bins   = [0, 20, 40, 60, 80, 100]
        score_labels = ["0-20", "21-40", "41-60", "61-80", "81-100"]
        score_dist   = pd.cut(df["score"], bins=score_bins, labels=score_labels).value_counts().sort_index()
        module_avg   = df.groupby("code_module")["score"].mean().round(2).to_dict()

        history_weak   = sum(1 for h in history if h["prediction"] == 1)
        history_strong = sum(1 for h in history if h["prediction"] == 0)

        return jsonify({
            "total_records":        total,
            "weak_students":        weak_count,
            "strong_students":      strong_count,
            "avg_score":            avg_score,
            "score_distribution":   score_dist.to_dict(),
            "module_avg_scores":    module_avg,
            "predictions_made":     len(history),
            "predictions_weak":     history_weak,
            "predictions_strong":   history_strong
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
