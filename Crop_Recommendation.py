from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# ✅ Load model
with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

# ✅ Load label encoder
with open('encoder2.pkl', 'rb') as f:
    encoder = pickle.load(f)

# ✅ Define categories
soil_types = [
    'Alluvial', 'Black Soil', 'Clay', 'Clay Loam',
    'Coastal Sandy', 'Loamy', 'Red Soil', 'Sandy', 'Sandy Loam'
]

season_types = ['Kharif', 'Rabi', 'Zayad']


@app.route('/')
def home():
    return "Crop Recommendation API is running 🚀"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # numeric inputs
        input_data = {
            'nitrogen': data['nitrogen'],
            'phosphorus': data['phosphorus'],
            'potassium': data['potassium'],
            'ph': data['ph'],
            'temperature': data['temperature'],
            'humidity': data['humidity'],
            'rainfall': data['rainfall']
        }

        # soil encoding
        for soil in soil_types:
            input_data[f'soil_{soil}'] = 1 if data['soil'] == soil else 0

        # season encoding
        for season in season_types:
            input_data[f'season_{season}'] = 1 if data['season'] == season else 0

        df = pd.DataFrame([input_data])
        print("Input DataFrame:")
        print(df.head())

        # prediction
        prediction = model.predict(df)
        probs = model.predict_proba(df)[0]

        # main crop
        crop_name = encoder.inverse_transform(prediction)[0]
        probability = round(float(max(probs)), 4)

        # top 3
        import numpy as np
        top3_idx = np.argsort(probs)[-3:][::-1]

        top3 = []
        for i in top3_idx:
            crop = encoder.inverse_transform([i])[0]
            top3.append({
                "crop": crop,
                "probability": round(float(probs[i]), 4)
            })

        return jsonify({
            "best_crop": crop_name,
            "probability": probability,
            "top_3_recommendations": top3
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)