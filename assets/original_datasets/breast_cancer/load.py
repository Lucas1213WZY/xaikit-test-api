import pandas as pd
import numpy as np
from sklearn import preprocessing
from datasets.tabular_dataset import TabularDataset
from ucimlrepo import fetch_ucirepo

def load_data(**kwargs):
    # Fetch the dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    
    # Data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    
    # Metadata (just for inspection, not used in processing)
    print(breast_cancer_wisconsin_diagnostic.metadata)
    print(breast_cancer_wisconsin_diagnostic.variables)
    
    # Target name
    target_name = "Diagnosis"

    # Ensure binary classification (if it's not already binary)
    unique_classes = y[target_name].unique()
    if len(unique_classes) != 2:
        raise ValueError("The target variable is not binary.")

    # Relabel the target if necessary
    y[target_name] = y[target_name].apply(lambda x: 0 if x == unique_classes[0] else 1)

    # Convert categorical features to numerical (if any)
    categorical_features = []  # Assuming no categorical features in this dataset
    categorical_feature_options = {}
    X_numpy = X.values

    for feature in categorical_features:
        le = preprocessing.LabelEncoder()
        valid_data = X.iloc[:, feature].fillna('Missing')
        le.fit(valid_data)
        transformed_data = le.transform(valid_data)
        try:
            missing_label_index = le.transform(['Missing'])[0]
            transformed_data = np.where(transformed_data == missing_label_index, np.nan, transformed_data)
        except:
            pass
        X_numpy[:, feature] = transformed_data
        classes_without_missing = np.delete(le.classes_, np.argwhere(le.classes_ == 'Missing'))
        categorical_feature_options[feature] = list(classes_without_missing)

    X_numpy = X_numpy.astype(float)
    y_numpy = y.values.squeeze()

    feature_names = X.columns.tolist()

    # Create the tabular dataset
    dataset = TabularDataset(X_numpy, y_numpy, feature_names=feature_names,
                             target_name=target_name, target_options=["Benign", "Malignant"],
                             categorical_feature_options=categorical_feature_options,
                             dataset_name="breast_cancer")

    return dataset
