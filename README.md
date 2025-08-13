🧠 Tariff-Fashion ML Module
A Streamlit-based ML app to predict fashion product prices & suggest optimal tariff brackets.
Built as part of 🌟 GirlScript Summer of Code 2025 (GSSoC'25).
📌 Overview
.Tariff-Fashion is a machine learning-powered web app that:
.Predicts the price of a fashion product based on its features.
.Suggests the tariff bracket to classify the product for optimal pricing.

Provides an easy-to-use Streamlit interface for predictions.
📊 Features
✅ ML-based price prediction using scikit-learn
✅ Automatic tariff bracket classification
✅ Clean Streamlit UI for quick usage
✅ Deployable on Streamlit Cloud in seconds

🛠 Tech Stack
.Language: Python 3.9+
.Framework: Streamlit
.ML Library: scikit-learn
.Model Storage: joblib

📂 Folder Structure
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



🤝 Contributing
1.We ❤️ contributions!

2.Fork the repo

3.Create a new branch (feature/new-feature)

4.Commit your changes
git status

5.Push to your fork

6.Open a Pull Request

Check out our CONTRIBUTING.md for more details.

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.