# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# ✅ Enable full CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# =========================
# 🔹 Load model and metadata
# =========================
try:
    print("DEBUG: Loading model and class data...")
    model = tf.keras.models.load_model("workout_model.h5")
    label_classes = np.load("label_classes.npy", allow_pickle=True)
    workout_type_classes = np.load("workout_type_classes.npy", allow_pickle=True)
    print("DEBUG: Model and class data loaded successfully.")
except Exception as e:
    print("ERROR: Failed to load model or class files:", e)
    raise e

# =========================
# 🔹 Compute mean & std from dataset
# =========================
try:
    data = pd.read_csv("workout_data.csv")
    cal_mean = data["avg_calories"].mean()
    cal_std = data["avg_calories"].std()
    print(f"DEBUG: Mean={cal_mean:.2f}, Std={cal_std:.2f}")
except Exception as e:
    print("ERROR: Failed to load workout_data.csv:", e)
    raise e


# =========================
# 🔹 Prediction endpoint
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_json = request.get_json()
        print("DEBUG: Received JSON input ->", data_json)

        if not data_json or "avg_calories" not in data_json or "workout_type" not in data_json:
            print("DEBUG: Missing required fields in JSON.")
            return jsonify({"error": "avg_calories and workout_type required"}), 400

        # Standardize avg_calories
        avg_calories = (float(data_json["avg_calories"]) - cal_mean) / cal_std
        print("DEBUG: Standardized avg_calories ->", avg_calories)

        # Encode workout_type with smoothing
        workout_onehot = np.zeros(len(workout_type_classes))
        workout_type_input = data_json["workout_type"].strip().lower()
        lc_classes = [w.lower() for w in workout_type_classes]

        if workout_type_input in lc_classes:
            idx = lc_classes.index(workout_type_input)
            workout_onehot[idx] = 0.9  # strong signal for known type
            print(f"DEBUG: Matched workout_type '{workout_type_input}' at index {idx}")
        else:
            workout_onehot += 0.1 / len(workout_type_classes)
            print(f"DEBUG: Unknown workout_type '{workout_type_input}', applied smoothing.")

        # Combine features
        features = np.concatenate([[avg_calories * 2], workout_onehot]).reshape(1, -1)
        print("DEBUG: Features passed to model ->", features)

        # Predict
        pred_probs = model.predict(features)
        pred_idx = np.argmax(pred_probs)
        predicted_goal = label_classes[pred_idx]

        print("DEBUG: Model prediction probabilities ->", pred_probs)
        print("DEBUG: Predicted goal ->", predicted_goal)

        return jsonify({"predicted_goal": predicted_goal})

    except Exception as e:
        print("DEBUG: Exception during prediction ->", e)
        return jsonify({"error": str(e)}), 500


# =========================
# 🔹 Run app
# =========================
if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)
