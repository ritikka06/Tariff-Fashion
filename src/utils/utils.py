import joblib

def load_model(path):
    return joblib.load(path)

def predict_price(model, input_df):
    return model.predict(input_df)[0]