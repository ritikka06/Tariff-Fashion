import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Tariff_fashion_cleaned.csv")

# Rename columns for convenience
df.rename(columns={
    "Product Type": "Category",
    "Brand Name": "Brand",
    "Price After Tariff": "Price",
    "Price Before Tariff": "BaseCost"
}, inplace=True)

# Select only available features
features = ["Category", "Brand", "BaseCost"]

# Drop NA rows for these columns and target
df.dropna(subset=features + ["Price"], inplace=True)

X = df[features]
y = df["Price"]

# Define categorical and numerical columns
categorical_cols = ["Category", "Brand"]
numerical_cols = ["BaseCost"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
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
joblib.dump(model_pipeline, "model/predictor.pkl")
print("Model saved to model/ppredictor.pkl")
