import pandas as pd
from datasets.tabular_dataset import TabularDataset
import os

def load_data(**kwargs):
    # fetch dataset
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "winequality-red.csv")
    wine_quality = pd.read_csv(csv_path)

    target_name = "quality"

    
    # Separate the target out and binarize it by converting >=7 to 1 <7 to 0
    y = wine_quality[target_name]
    y = (y >= 7).astype(int)
    X = wine_quality.drop(columns=[target_name])


    map_feature_names = {
    "alcohol" : "Alcohol",
    "sulphates": "Sulphates",
    "total sulfur dioxide": "SO2",  # "SO2" is a common abbreviation for sulfur dioxide
    "volatile acidity": "Vinegar Taint"  # Shortened from "volatile acidity"
    }

    # Rename feature columns based on the mapping
    X.rename(columns=map_feature_names, inplace=True)

    feature_names = [col for col in X.columns if col != target_name]


    X_numpy = X.dropna().values
    y_numpy = y.loc[X.index].values

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
        target_name=target_name, target_options=["Type 1", "Type 2"], categorical_feature_options=categorical_feature_options, 
        dataset_name="wine_quality")
    return dataset