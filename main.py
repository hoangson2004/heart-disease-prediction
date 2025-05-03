import joblib
from heart_disease_predictor.modeling.dataset import load_data, split_data
from heart_disease_predictor.modeling.predict import predict
from heart_disease_predictor.modeling.train import train_model

def main():

    disease_df = load_data()
    X_train, X_test, y_train, y_test = split_data(disease_df)

    train_model(X_train, y_train)

    model = joblib.load("models/logreg_model.pkl")

    y_pred = model.predict(X_test)

    predict(y_test, y_pred)

if __name__ == "__main__":
    main()