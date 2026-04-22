# heart-disease-prediction-naive-bayes
 Heart Disease Prediction using Naive Bayes
📌 Overview

This project demonstrates a machine learning approach to predict the presence of heart disease using the Gaussian Naive Bayes algorithm. The model is trained on a dataset containing medical attributes and evaluated using standard classification metrics.

⚙️ Technologies Used
Python
Pandas
Scikit-learn
📊 Dataset

The dataset (heart.csv) contains various medical features such as:

Age
Sex
Chest pain type
Blood pressure
Cholesterol levels
Maximum heart rate
And more...

The target variable:

0 → No Heart Disease
1 → Presence of Heart Disease
🚀 Project Workflow
Data Loading using Pandas
Data Preprocessing (feature-target split)
Train-Test Split (80% training, 20% testing)
Model Training using Gaussian Naive Bayes
Model Evaluation using:
Accuracy
Precision
Recall
🧠 Model Used
Gaussian Naive Bayes
Suitable for continuous numerical features
Based on probability and Bayes' theorem
📈 Evaluation Metrics

The model is evaluated using:

Accuracy Score
Precision Score
Recall Score

Example output:

Recall Score: 0.84
Accuracy Score: 0.86
Precision Score: 0.9
🛠️ Installation & Setup
Clone the repository:
git clone https://github.com/your-username/heart-disease-prediction-naive-bayes.git
cd heart-disease-prediction-naive-bayes
Install dependencies:
pip install pandas scikit-learn
Run the script:
python naive_bayes.py
📂 Project Structure
├── heart.csv
├── naive_bayes.py
└── README.md
📌 Key Learnings
Implementation of Naive Bayes for classification
Working with real-world healthcare data
Model evaluation using multiple metrics
🔮 Future Improvements
Add data visualization
Perform feature engineering
Compare with other ML models (e.g. Random Forest)
