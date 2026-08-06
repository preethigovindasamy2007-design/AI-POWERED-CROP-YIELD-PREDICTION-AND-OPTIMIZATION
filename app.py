
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        rainfall = float(request.form["rainfall"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        area = float(request.form["area"])

        new_data = pd.DataFrame({
            "Rainfall": [rainfall],
            "Temperature": [temperature],
            "Humidity": [humidity],
            "Area": [area]
        })

        prediction = model.predict(new_data)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)