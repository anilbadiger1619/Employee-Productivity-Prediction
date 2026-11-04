from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model/productivity_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    working_hours = float(request.form["working_hours"])
    attendance = float(request.form["attendance"])
    tasks_completed = float(request.form["tasks_completed"])
    experience = float(request.form["experience"])
    overtime = float(request.form["overtime"])
    leave_days = float(request.form["leave_days"])
    satisfaction = float(request.form["satisfaction"])
    previous_performance = float(request.form["previous_performance"])

    # Create input
    input_data = np.array([[
        working_hours,
        attendance,
        tasks_completed,
        experience,
        overtime,
        leave_days,
        satisfaction,
        previous_performance
    ]])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Keep score between 0 and 100
    prediction = max(0, min(100, prediction))

    prediction = round(prediction, 2)

    # Productivity category
    if prediction >= 80:
        status = "High Productivity"
    elif prediction >= 60:
        status = "Medium Productivity"
    else:
        status = "Low Productivity"

    return render_template(
        "result.html",
        prediction=prediction,
        status=status
    )


if __name__ == "__main__":
    app.run(debug=True)