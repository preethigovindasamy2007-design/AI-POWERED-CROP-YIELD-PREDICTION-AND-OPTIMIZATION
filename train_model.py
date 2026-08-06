import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("dataset/crop_yield.csv")

print("Crop Yield Dataset")
print(data)

# Features
X = data[["Rainfall", "Temperature", "Humidity", "Area"]]

# Target
y = data["Production"]

# Create AI Model
model = RandomForestRegressor()

# Train Model
model.fit(X, y)

print("\nAI Model Trained Successfully!")

# Test Prediction
new_data = pd.DataFrame({
    "Rainfall": [900],
    "Temperature": [29],
    "Humidity": [80],
    "Area": [10]
})

prediction = model.predict(new_data)

print("Predicted Crop Yield:", prediction[0])
joblib.dump(model, "model.pkl")
print("Model saved successfully!")