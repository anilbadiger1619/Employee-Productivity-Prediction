import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load dataset
data = pd.read_csv("dataset/employee_productivity.csv")

# Input features
X = data[
    [
        "Working_Hours",
        "Attendance",
        "Tasks_Completed",
        "Experience_Years",
        "Overtime_Hours",
        "Leave_Days",
        "Job_Satisfaction",
        "Previous_Performance"
    ]
]

# Target
y = data["Productivity_Score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Test
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("====================================")
print("EMPLOYEE PRODUCTIVITY MODEL")
print("====================================")
print("Model trained successfully!")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/productivity_model.pkl")

print("Model saved successfully!")
print("Location: model/productivity_model.pkl")