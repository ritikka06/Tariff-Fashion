# train_model.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error


df = pd.read_csv("fashion_dataset.csv")

X = df.drop("Price", axis=1)
y = df["Price"]

# Define categorical and numerical columns
categorical_cols = ['Category', 'Brand', 'Material', 'Region']
numerical_cols = ['BaseCost', 'Weight', 'Rating']

# ColumnTransformer for encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # keeps numerical features
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")

os.makedirs("model", exist_ok=True)

joblib.dump(model_pipeline, "model/price_predictor.pkl")
print("Model saved to model/price_predictor.pkl")
