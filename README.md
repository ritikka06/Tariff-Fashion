# 🧠 Tariff-Fashion ML Module

This Streamlit app predicts the price of a fashion product based on its features and suggests an optimal tariff bracket. Built as part of **GSSoC 2025**.

## 📊 Features

- Price prediction using trained ML model
- Tariff bracket suggestion (based on predicted price)
- Clean Streamlit UI
- Easy deployment on Streamlit Cloud

## 🛠 Tech Stack

- Python
- scikit-learn
- Streamlit
- joblib

## 📁 Folder Structure

Tariff-Fashion/
│
├── src/
│   ├── data/              
│   │   ├── fashion_dataset.csv
│   │   ├── Tariff_fashion_cleaned.csv
│   │
│   ├── eda/               
│   │   └── tariff_fashion.ipynb
│   │
│   ├── models/            
│   │   ├── train_model.py
│   │   └── model/         
│   │       └── predictor.pkl   
│
├── app/                   
│   ├── app.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore

