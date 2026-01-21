from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Paths for the model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'wine_cultivar_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

# Load files
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Create array of the 6 inputs
        raw_data = np.array([[
            float(data['alcohol']), float(data['malic_acid']), 
            float(data['ash']), float(data['alcalinity_of_ash']), 
            float(data['magnesium']), float(data['flavanoids'])
        ]])
        
        # Scale then Predict
        scaled_data = scaler.transform(raw_data)
        prediction = model.predict(scaled_data)[0]
        
        # Map 0, 1, 2 to Cultivar names
        names = ["Cultivar 1", "Cultivar 2", "Cultivar 3"]
        return jsonify({'result': names[prediction]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)