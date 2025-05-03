import joblib
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    joblib.dump(logreg, "models/logreg_model.pkl")
    return logreg
