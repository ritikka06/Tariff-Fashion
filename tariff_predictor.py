import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class FashionTariffPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = ['category', 'material', 'origin_country', 'destination_country', 
                               'value_usd', 'weight_kg', 'brand_tier']
        
    def generate_sample_data(self, n_samples=1000):
        """Generate sample fashion tariff data for training"""
        np.random.seed(42)
        
        categories = ['Clothing', 'Footwear', 'Accessories', 'Bags', 'Jewelry', 'Textiles']
        materials = ['Cotton', 'Polyester', 'Leather', 'Silk', 'Wool', 'Synthetic', 'Metal', 'Plastic']
        countries = ['China', 'India', 'Vietnam', 'Bangladesh', 'Turkey', 'Italy', 'USA', 'Germany']
        brand_tiers = ['Luxury', 'Premium', 'Mid-range', 'Budget']
        
        data = []
        for _ in range(n_samples):
            category = np.random.choice(categories)
            material = np.random.choice(materials)
            origin = np.random.choice(countries)
            destination = np.random.choice(countries)
            brand_tier = np.random.choice(brand_tiers)
            
            # Generate realistic values based on category
            if category == 'Jewelry':
                value = np.random.uniform(50, 5000)
                weight = np.random.uniform(0.01, 0.5)
            elif category == 'Footwear':
                value = np.random.uniform(20, 800)
                weight = np.random.uniform(0.3, 1.5)
            elif category == 'Bags':
                value = np.random.uniform(15, 2000)
                weight = np.random.uniform(0.2, 2.0)
            else:
                value = np.random.uniform(10, 1000)
                weight = np.random.uniform(0.1, 2.0)
            
            # Calculate tariff based on complex rules (simulating real-world scenarios)
            base_rate = 0.05  # 5% base rate
            
            # Category-based adjustments
            category_multipliers = {
                'Clothing': 1.0, 'Footwear': 1.2, 'Accessories': 0.8,
                'Bags': 1.1, 'Jewelry': 0.6, 'Textiles': 1.3
            }
            
            # Material-based adjustments
            material_multipliers = {
                'Cotton': 1.0, 'Polyester': 1.1, 'Leather': 1.3,
                'Silk': 0.9, 'Wool': 1.2, 'Synthetic': 1.15,
                'Metal': 0.8, 'Plastic': 1.25
            }
            
            # Country-based adjustments (trade agreements, etc.)
            country_adjustments = {
                'China': 1.4, 'India': 1.1, 'Vietnam': 1.0, 'Bangladesh': 0.9,
                'Turkey': 1.0, 'Italy': 0.7, 'USA': 0.5, 'Germany': 0.6
            }
            
            # Brand tier adjustments
            brand_adjustments = {
                'Luxury': 0.8, 'Premium': 0.9, 'Mid-range': 1.0, 'Budget': 1.2
            }
            
            tariff_rate = (base_rate * 
                          category_multipliers.get(category, 1.0) *
                          material_multipliers.get(material, 1.0) *
                          country_adjustments.get(origin, 1.0) *
                          brand_adjustments.get(brand_tier, 1.0))
            
            # Add some randomness
            tariff_rate *= np.random.uniform(0.8, 1.2)
            tariff_rate = max(0.01, min(0.5, tariff_rate))  # Cap between 1% and 50%
            
            tariff_amount = value * tariff_rate
            
            data.append({
                'category': category,
                'material': material,
                'origin_country': origin,
                'destination_country': destination,
                'value_usd': value,
                'weight_kg': weight,
                'brand_tier': brand_tier,
                'tariff_rate': tariff_rate,
                'tariff_amount': tariff_amount
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data for training or prediction"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['category', 'material', 'origin_country', 'destination_country', 'brand_tier']
        
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                # Handle unseen categories
                df_processed[col] = df_processed[col].apply(
                    lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown'
                )
                if 'Unknown' not in self.label_encoders[col].classes_:
                    # Add 'Unknown' to classes
                    self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'Unknown')
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def train(self, df=None):
        """Train the tariff prediction model"""
        if df is None:
            print("Generating sample training data...")
            df = self.generate_sample_data(1000)
        
        print(f"Training on {len(df)} samples...")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, is_training=True)
        
        # Prepare features and target
        X = df_processed[self.feature_columns]
        y = df_processed['tariff_amount']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return mae, r2
    
    def predict(self, item_data):
        """Predict tariff for a single item or list of items"""
        if isinstance(item_data, dict):
            df = pd.DataFrame([item_data])
        else:
            df = pd.DataFrame(item_data)
        
        # Preprocess
        df_processed = self.preprocess_data(df, is_training=False)
        X = df_processed[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        if len(predictions) == 1:
            return predictions[0]
        return predictions
    
    def save_model(self, filepath='tariff_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='tariff_model.pkl'):
        """Load a trained model and preprocessors"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {filepath}")

def main():
    """Example usage of the FashionTariffPredictor"""
    predictor = FashionTariffPredictor()
    
    # Train the model
    mae, r2 = predictor.train()
    
    # Save the model
    predictor.save_model()
    
    # Example predictions
    sample_items = [
        {
            'category': 'Clothing',
            'material': 'Cotton',
            'origin_country': 'China',
            'destination_country': 'USA',
            'value_usd': 50.0,
            'weight_kg': 0.5,
            'brand_tier': 'Mid-range'
        },
        {
            'category': 'Footwear',
            'material': 'Leather',
            'origin_country': 'Italy',
            'destination_country': 'USA',
            'value_usd': 200.0,
            'weight_kg': 1.0,
            'brand_tier': 'Luxury'
        }
    ]
    
    print("\nSample Predictions:")
    for i, item in enumerate(sample_items):
        predicted_tariff = predictor.predict(item)
        print(f"Item {i+1}: ${predicted_tariff:.2f} tariff (Value: ${item['value_usd']})")

if __name__ == "__main__":
    main()
