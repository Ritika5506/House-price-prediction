from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

# Configure static folder
app.static_folder = 'static'
app.static_url_path = '/static'

# Get the project root directory (parent of app folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load trained model
model = pickle.load(open(os.path.join(PROJECT_ROOT, "models/house_price_model.pkl"), "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    rooms = float(request.form["rooms"])
    lstat = float(request.form["lstat"])
    ptratio = float(request.form["ptratio"])

    features = np.array([[rooms, lstat, ptratio]])

    prediction = model.predict(features)

    return render_template("index.html", prediction_text=f"Predicted House Price: {prediction[0]}")

if __name__ == "__main__":
    app.run(debug=True)