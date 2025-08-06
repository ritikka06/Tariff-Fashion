# Fashion Tariff Predictor 🧥💰

An AI-powered machine learning application that predicts import/export tariffs for fashion items using Python, scikit-learn, and Flask.

## Features

- **ML-Powered Predictions**: Uses Random Forest algorithm to predict tariffs based on item characteristics
- **Web Interface**: Beautiful, responsive web UI for easy tariff calculations
- **Comprehensive Analysis**: Data visualization and insights dashboard
- **Batch Processing**: Support for multiple item predictions
- **Real-time Results**: Instant tariff calculations with detailed breakdowns

## Technology Stack

- **Backend**: Python, Flask, scikit-learn
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **ML Libraries**: pandas, numpy, matplotlib, seaborn, plotly
- **Data Processing**: Label encoding, feature scaling, model persistence

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Tariff-Fashion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

### Web Interface
1. Fill in the fashion item details (category, material, origin, etc.)
2. Click "Predict Tariff" to get instant results
3. View tariff amount, rate percentage, and item value

### Command Line
```python
from tariff_predictor import FashionTariffPredictor

predictor = FashionTariffPredictor()
predictor.train()  # Train the model

# Predict tariff for an item
item = {
    'category': 'Clothing',
    'material': 'Cotton',
    'origin_country': 'China',
    'destination_country': 'USA',
    'value_usd': 50.0,
    'weight_kg': 0.5,
    'brand_tier': 'Mid-range'
}

tariff = predictor.predict(item)
print(f"Predicted tariff: ${tariff:.2f}")
```

### Data Analysis
Run comprehensive analysis:
```bash
python data_analysis.py
```

This generates:
- Static visualizations (`tariff_analysis.png`)
- Interactive dashboard (`interactive_dashboard.html`)
- Key insights (`tariff_insights.txt`)

## Model Features

The ML model considers these factors:
- **Category**: Clothing, Footwear, Accessories, Bags, Jewelry, Textiles
- **Material**: Cotton, Polyester, Leather, Silk, Wool, Synthetic, Metal, Plastic
- **Origin Country**: China, India, Vietnam, Bangladesh, Turkey, Italy, USA, Germany
- **Destination Country**: Same options as origin
- **Value (USD)**: Item monetary value
- **Weight (kg)**: Item weight
- **Brand Tier**: Luxury, Premium, Mid-range, Budget

## API Endpoints

- `GET /` - Web interface
- `POST /predict` - Single item prediction
- `POST /batch_predict` - Multiple items prediction
- `GET /model_info` - Model information and feature importance

## Project Structure

```
Tariff-Fashion/
├── app.py                 # Flask web application
├── tariff_predictor.py    # ML model implementation
├── data_analysis.py       # Data analysis and visualization
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface template
├── tariff_model.pkl      # Trained model (generated)
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - feel free to use this project for educational or commercial purposes.

## Future Enhancements

- [ ] Integration with real tariff databases
- [ ] Support for more countries and materials
- [ ] Historical tariff trend analysis
- [ ] API rate limiting and authentication
- [ ] Docker containerization
- [ ] Cloud deployment options