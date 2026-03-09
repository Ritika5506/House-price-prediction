from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load trained model
model = pickle.load(open("models/house_price_model.pkl", "rb"))

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