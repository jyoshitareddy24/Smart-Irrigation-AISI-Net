📌 Project Title
Intelligent Deep Learning-Based Smart Irrigation System

🌱 Problem Statement
Efficient water management in agriculture is essential to prevent water wastage and improve crop productivity. Traditional irrigation systems lack intelligent decision-making capabilities. This project proposes a deep learning-based smart irrigation system that predicts irrigation requirements based on environmental factors.

📊 Dataset Description
A synthetic agricultural dataset was generated containing:
- Temperature
- Humidity
- Soil Moisture
- Rainfall
- Irrigation Requirement (0 or 1)

Total Samples: 1000
Training Set: 800
Testing Set: 200

🤖 Models Implemented
1. Logistic Regression
2. Random Forest
3. Proposed Model: AISI-Net

🧠 Proposed Model – AISI-Net
AISI-Net (Adaptive Intelligent Smart Irrigation Network) is a deep learning model consisting of:
- Dense Layer (64 neurons, ReLU)
- Batch Normalization
- Dropout (0.3)
- Dense Layer (32 neurons, ReLU)
- Dropout (0.2)
- Output Layer (Sigmoid)

This architecture improves generalization and reduces overfitting.

📈 Results
Logistic Regression Accuracy: 96.5%
Random Forest Accuracy: 100%
AISI-Net Accuracy: 97.5%

📉 Performance Metrics (AISI-Net)
- Accuracy: ~97%
- Precision: 1.00
- Recall: 0.67
- F1-score: 0.80

📊 Visualizations
- Confusion Matrix (See Screenshots folder)
- Accuracy Comparison Graph (See Screenshots folder)

🛠 Technologies Used
- Python 3.8
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Matplotlib
- Seaborn

🚀 How to Run
1. Open Command Prompt
2. Navigate to project folder
3. Run: python main.py
2. Navigate to project folder

3. Run: python main.py
