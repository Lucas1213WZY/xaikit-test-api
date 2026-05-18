from ucimlrepo import fetch_ucirepo 
from sklearn import preprocessing
from datasets.tabular_dataset import TabularDataset
import numpy as np
import pandas as pd

def load_data(**kwargs):
    # Fetch dataset
    auto_mpg = fetch_ucirepo(id=9)
    print("Retrieved online dataset")

    # Data (as pandas dataframes)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets

    target_name = "mpg"

    # Clean the dataset
    X.replace('?', np.nan, inplace=True)
    X.dropna(inplace=True)
    
    # Separate target column
    y = X[target_name].astype(float)
    X = X.drop(columns=[target_name])

    # Convert categorical features to numerical
    categorical_features = ["origin", "car name"]
    categorical_feature_options = {}
    for feature in categorical_features:
        le = preprocessing.LabelEncoder()
        valid_data = X[feature].fillna('Missing')
        le.fit(valid_data)
        transformed_data = le.transform(valid_data)
        try:
            missing_label_index = le.transform(['Missing'])[0]
            transformed_data = np.where(transformed_data == missing_label_index, np.nan, transformed_data)
        except:
            pass
        X[feature] = transformed_data
        classes_without_missing = np.delete(le.classes_, np.argwhere(le.classes_ == 'Missing'))
        categorical_feature_options[X.columns.get_loc(feature)] = list(classes_without_missing)

    # Convert to numpy arrays
    X_numpy = X.values.astype(float)
    y_numpy = y.values

    # Binary classification based on median MPG
    median_mpg = np.median(y_numpy)
    y_binary = np.where(y_numpy >= median_mpg, 1, 0)

    # Create the tabular dataset
    feature_names = X.columns.tolist()
    dataset = TabularDataset(X_numpy, y_binary, feature_names=feature_names,
                             target_name=target_name, target_options=["Below Median MPG", "Above Median MPG"],
                             categorical_feature_options=categorical_feature_options, dataset_name="auto_mpg")

    # Optionally, remove or use specific features
    # dataset = dataset.remove_specific_features([...])
    # dataset = dataset.use_specific_features([...])

    return dataset

# Load the dataset
dataset = load_data()
print(dataset)
