from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from tariff_predictor import FashionTariffPredictor

app = Flask(__name__)

# Initialize the predictor
predictor = FashionTariffPredictor()

# Train or load model on startup
model_path = 'tariff_model.pkl'
if os.path.exists(model_path):
    try:
        predictor.load_model(model_path)
        print("Model loaded successfully!")
    except:
        print("Error loading model, training new one...")
        predictor.train()
        predictor.save_model(model_path)
else:
    print("No existing model found, training new one...")
    predictor.train()
    predictor.save_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_tariff():
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['category', 'material', 'origin_country', 'destination_country', 
                          'value_usd', 'weight_kg', 'brand_tier']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Convert numeric fields
        try:
            data['value_usd'] = float(data['value_usd'])
            data['weight_kg'] = float(data['weight_kg'])
        except ValueError:
            return jsonify({'error': 'Value and weight must be numeric'}), 400
        
        # Make prediction
        predicted_tariff = predictor.predict(data)
        tariff_rate = predicted_tariff / data['value_usd'] if data['value_usd'] > 0 else 0
        
        return jsonify({
            'predicted_tariff': round(predicted_tariff, 2),
            'tariff_rate': round(tariff_rate * 100, 2),  # Convert to percentage
            'item_value': data['value_usd'],
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        items = data.get('items', [])
        
        if not items:
            return jsonify({'error': 'No items provided'}), 400
        
        predictions = []
        for item in items:
            try:
                predicted_tariff = predictor.predict(item)
                tariff_rate = predicted_tariff / item['value_usd'] if item['value_usd'] > 0 else 0
                
                predictions.append({
                    'item': item,
                    'predicted_tariff': round(predicted_tariff, 2),
                    'tariff_rate': round(tariff_rate * 100, 2)
                })
            except Exception as e:
                predictions.append({
                    'item': item,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': predictions,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/model_info')
def model_info():
    try:
        # Get feature importance if available
        feature_importance = []
        if hasattr(predictor.model, 'feature_importances_'):
            feature_importance = [
                {
                    'feature': feature,
                    'importance': round(importance, 3)
                }
                for feature, importance in zip(predictor.feature_columns, predictor.model.feature_importances_)
            ]
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({
            'feature_importance': feature_importance,
            'model_type': type(predictor.model).__name__,
            'features': predictor.feature_columns,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
