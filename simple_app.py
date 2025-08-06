from flask import Flask, render_template, request, jsonify
import random
import math

app = Flask(__name__)

class SimpleTariffPredictor:
    def __init__(self):
        # Simple rule-based tariff calculation
        self.category_rates = {
            'Clothing': 0.12,
            'Footwear': 0.15,
            'Accessories': 0.08,
            'Bags': 0.11,
            'Jewelry': 0.06,
            'Textiles': 0.13
        }
        
        self.material_multipliers = {
            'Cotton': 1.0,
            'Polyester': 1.1,
            'Leather': 1.3,
            'Silk': 0.9,
            'Wool': 1.2,
            'Synthetic': 1.15,
            'Metal': 0.8,
            'Plastic': 1.25
        }
        
        self.country_adjustments = {
            'China': 1.4,
            'India': 1.1,
            'Vietnam': 1.0,
            'Bangladesh': 0.9,
            'Turkey': 1.0,
            'Italy': 0.7,
            'USA': 0.5,
            'Germany': 0.6
        }
        
        self.brand_adjustments = {
            'Luxury': 0.8,
            'Premium': 0.9,
            'Mid-range': 1.0,
            'Budget': 1.2
        }
    
    def predict(self, item_data):
        """Simple rule-based tariff prediction"""
        base_rate = self.category_rates.get(item_data['category'], 0.1)
        
        # Apply multipliers
        material_mult = self.material_multipliers.get(item_data['material'], 1.0)
        country_mult = self.country_adjustments.get(item_data['origin_country'], 1.0)
        brand_mult = self.brand_adjustments.get(item_data['brand_tier'], 1.0)
        
        # Calculate final rate
        final_rate = base_rate * material_mult * country_mult * brand_mult
        
        # Add some randomness for realism
        final_rate *= random.uniform(0.9, 1.1)
        
        # Cap between 1% and 40%
        final_rate = max(0.01, min(0.4, final_rate))
        
        # Calculate tariff amount
        tariff_amount = item_data['value_usd'] * final_rate
        
        return tariff_amount

# Initialize predictor
predictor = SimpleTariffPredictor()

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
            'tariff_rate': round(tariff_rate * 100, 2),
            'item_value': data['value_usd'],
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/model_info')
def model_info():
    return jsonify({
        'model_type': 'Rule-based Calculator',
        'features': ['category', 'material', 'origin_country', 'destination_country', 'value_usd', 'weight_kg', 'brand_tier'],
        'feature_importance': [
            {'feature': 'category', 'importance': 0.25},
            {'feature': 'origin_country', 'importance': 0.20},
            {'feature': 'material', 'importance': 0.18},
            {'feature': 'brand_tier', 'importance': 0.15},
            {'feature': 'value_usd', 'importance': 0.12},
            {'feature': 'weight_kg', 'importance': 0.10}
        ],
        'success': True
    })

if __name__ == '__main__':
    print("🚀 Starting Fashion Tariff Predictor...")
    print("📊 Using rule-based calculation engine")
    print("🌐 Server will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
