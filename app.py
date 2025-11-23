from flask import Flask, request, jsonify, render_template
import joblib, numpy as np, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
model = joblib.load(MODEL_PATH)

FEATURES = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]

app = Flask(__name__, template_folder="templates")

@app.get("/health")
def health():
    return {"ok": True, "features": FEATURES}

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    x = data.get("features")
    if not x or len(x) != len(FEATURES):
        return jsonify(error=f"Expected 8 features: {FEATURES}"), 400
    X = np.array([x], dtype=float)
    yhat = model.predict(X).tolist()[0]
    return jsonify(prediction=yhat)

@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "GET":
        return render_template("main.html")
    vals = [float(request.form.get(f)) for f in FEATURES]
    pred = model.predict(np.array([vals])).tolist()[0]
    return render_template("main.html", prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)