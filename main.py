import numpy as np
import pandas as pd

# Set random seed
np.random.seed(42)

# Create synthetic smart irrigation dataset
data_size = 1000

temperature = np.random.uniform(20, 40, data_size)
humidity = np.random.uniform(30, 90, data_size)
soil_moisture = np.random.uniform(10, 60, data_size)
rainfall = np.random.uniform(0, 20, data_size)

# Irrigation rule
irrigation = ((soil_moisture < 30) & (rainfall < 5)).astype(int)

data = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Soil_Moisture': soil_moisture,
    'Rainfall': rainfall,
    'Irrigation': irrigation
})

print(data.head())
print("\nDataset Shape:", data.shape)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = data.drop('Irrigation', axis=1)
y = data['Irrigation']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print("\nLogistic Regression Accuracy:", lr_accuracy)

from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Build AISI-Net Model
dl_model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

dl_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = dl_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
dl_loss, dl_accuracy = dl_model.evaluate(X_test, y_test)

print("AISI-Net Accuracy:", dl_accuracy)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Predict for confusion matrix
dl_predictions = (dl_model.predict(X_test) > 0.5).astype(int)

cm = confusion_matrix(y_test, dl_predictions)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - AISI-Net")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Accuracy comparison plot
models = ['Logistic Regression', 'Random Forest', 'AISI-Net']
accuracies = [lr_accuracy, rf_accuracy, dl_accuracy]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.01)
plt.show()

from sklearn.metrics import classification_report

print("\nClassification Report for AISI-Net:\n")
print(classification_report(y_test, dl_predictions))

# Save model comparison results
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "AISI-Net"],
    "Accuracy": [lr_accuracy, rf_accuracy, dl_accuracy]
})

results.to_csv("model_results.csv", index=False)

print("\nModel results saved to model_results.csv")