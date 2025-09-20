from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import tensorflow as tf

# ================================
# Load saved models & preprocessors
# ================================
os.makedirs("models", exist_ok=True)

try:
    # Load the Keras model for crop prediction
    crop_model = tf.keras.models.load_model('models/crop_recommendation_model.h5')
    
    # Load the scikit-learn preprocessors for the crop model
    crop_scaler = joblib.load('models/crop_scaler.pkl')
    crop_encoder = joblib.load('models/crop_label_encoder.pkl')
    
    print("✅ Models loaded successfully.")
    
except FileNotFoundError as e:
    print(f"❌ Error: A model file was not found. Please ensure all model files are in the 'models' directory. Details: {e}")
    exit()

# ================================
# Fertilizer Recommendation Map
# ================================
# This map is based on the logic from your `crop_fertilizer.ipynb` file.
fertilizer_map = {
    'rice': 'Urea', 'maize': 'Urea', 'chickpea': 'DAP', 'kidneybeans': 'Urea', 
    'pigeonpeas': 'DAP', 'mothbeans': 'Urea', 'mungbean': 'DAP', 'blackgram': 'DAP',
    'lentil': 'DAP', 'pomegranate': 'NPK', 'banana': 'Urea', 'mango': 'Urea', 
    'grapes': 'NPK', 'watermelon': 'NPK', 'muskmelon': 'NPK', 'apple': 'NPK',
    'orange': 'NPK', 'papaya': 'Urea', 'coconut': 'NPK', 'cotton': 'NPK',
    'jute': 'Urea', 'coffee': 'DAP'
}

# ================================
# Flask App and Routes
# ================================
app = Flask(__name__)
CORS(app)

# List of features for crop prediction in the correct order
CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

@app.route("/")
def index():
    return jsonify({
        "service": "Crop & Fertilizer Recommendation API",
        "status": "running",
        "endpoint": "/predict"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Step 1: Predict the crop
        X = [float(data[f]) for f in CROP_FEATURES]
        X = np.array(X).reshape(1, -1)
        X_scaled = crop_scaler.transform(X)
        prediction_idx = np.argmax(crop_model.predict(X_scaled, verbose=0), axis=1)[0]
        predicted_crop = crop_encoder.inverse_transform([prediction_idx])[0]

        # Step 2: Recommend the fertilizer based on the predicted crop
        recommended_fertilizer = fertilizer_map.get(predicted_crop, 'General Urea/DAP')

        return jsonify({
            "crop_recommended": predicted_crop,
            "fertilizer_recommended": recommended_fertilizer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)