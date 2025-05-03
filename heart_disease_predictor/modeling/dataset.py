import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_data():
    disease_df = pd.read_csv("data/raw/framingham.csv")
    disease_df.drop("education", axis=1, inplace=True)
    disease_df.rename(columns= {'male': 'Sex_male'}, inplace=True)
    disease_df.dropna(axis=0, inplace=True)
    print(disease_df.TenYearCHD.value_counts())
    return disease_df

def split_data(disease_df):
    X = np.array(disease_df[['age', 'Sex_male', 'currentSmoker', 'cigsPerDay', 'prevalentStroke',
                           'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'BMI',
                            'heartRate', 'glucose']])
    y = np.array(disease_df['TenYearCHD'])

    X = preprocessing.MinMaxScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test