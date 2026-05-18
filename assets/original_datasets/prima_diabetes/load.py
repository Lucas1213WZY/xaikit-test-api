import pandas as pd
from datasets.tabular_dataset import TabularDataset
import os

def load_data(**kwargs):
    # fetch dataset
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "data.csv")
    diabetes = pd.read_csv(csv_path)

    target_name = "Outcome"


    # Separate the target out and binarize it by converting >=7 to 1 <7 to 0
    y = diabetes[target_name]
    X = diabetes.drop(columns=[target_name])


    # Remove rows where 'Glucose' is 0
    mask = X['Glucose'] != 0
    X = X[mask]
    y = y[mask]

    map_feature_names = {
    "DiabetesPedigreeFunction" : "DPF",
    }

    # Rename feature columns based on the mapping
    X.rename(columns=map_feature_names, inplace=True)

    feature_names = [col for col in X.columns if col != target_name]
    
    X_numpy = X.dropna().values
    y_numpy = y.loc[X.index].values

    # one outlier value


    # Converts the data from strings into numerical data by counting all the labels
    categorical_features = {}
    categorical_feature_options = {}
    for feature in categorical_features:
        le = preprocessing.LabelEncoder()
        le.fit(X_numpy[:, feature])
        X_numpy[:, feature] = le.transform(X_numpy[:, feature])
        categorical_feature_options[feature] = le.classes_
    X_numpy = X_numpy.astype(float)



    # create the tabular dataset
    dataset = TabularDataset(X_numpy, y_numpy, feature_names=feature_names,
        target_name=target_name, target_options=["No diabetes", "Has diabetes"], categorical_feature_options=categorical_feature_options, 
        dataset_name="prima_diabetes")
    return dataset