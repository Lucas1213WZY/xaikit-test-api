from sklearn import preprocessing
from datasets.tabular_dataset import TabularDataset
import numpy as np
import pandas as pd
import os

def load_data(**kwargs):
    # Load the dataset from a local CSV file
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "data.csv")
    df = pd.read_csv(file_path, delimiter=';')  # Use semicolon as delimiter
    print("Loaded data from local CSV file")

    # Drop rows where 'gill-spacing' has NA values
    df = df.dropna(subset=['gill-spacing'])
    print(f"Remaining rows after dropping rows with NA in 'gill-spacing': {len(df)}")

    # Shortlist rows where 'cap-shape' is 'x' or 'f', and rename the column to 'Shape'
    df = df[df['cap-shape'].isin(['x', 'f'])]
    print(f"Remaining rows after filtering 'Shape' to 'x' or 'f': {len(df)}")

    # Rename values in the specified columns
    rename_mappings = {
        'cap-shape': {'x': 'convex', 'f': 'flat'},
        'has-ring': {'f': 'no', 't': 'yes'},
        'does-bruise-or-bleed': {'f': 'no', 't': 'yes'},
        'gill-spacing': {'c': 'close', 'd': 'far', 'f': 'none'}
    }
    for column, mapping in rename_mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)

    # Drop columns with missing values
    df.dropna(axis=1, inplace=True)
    print(f"Remaining columns after dropping those with missing values: {df.columns.tolist()}")

    # Separate the target column
    target_name = "class"
    X = df.drop(columns=[target_name])
    y = df[target_name]

    # Map feature names to simplified names
    map_feature_names = {
        "spore-print-color": "spore-color",
        "gill-attachment": "gill-attach",
        "does-bruise-or-bleed": "Bruises",
        "cap-diameter": "Cap Diameter",
        "stem-height": "Height",
        "stem-width": "Width",
        "has-ring": "Ring",
        'gill-spacing': "Gill Spacing",
        'veil-type': "Veil Type",
        'cap-shape': 'Shape',
    }
    X.rename(columns=map_feature_names, inplace=True)

    feature_names = [col for col in X.columns if col != target_name]
    X_numpy = X.values
    y_numpy = y.values

    # Convert categorical features to numerical values
    categorical_features = ["Shape", "cap-surface", "cap-color", "Bruises", "gill-attach", "Gill Spacing",
                            "gill-color", "stem-root", "stem-surface", "stem-color", "Veil Type", "veil-color",
                            "Ring", "ring-type", "spore-color", "habitat", "season"]
    categorical_features = [X.columns.get_loc(feature) for feature in categorical_features if feature in X.columns]

    categorical_feature_options = {}
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

    # Preprocess the target variable
    y_numerical = np.where(y_numpy == 'p', 0, 1)
    target_name = "poisonous"

    # Create the tabular dataset
    dataset = TabularDataset(X_numpy, y_numerical, feature_names=feature_names,
                             target_name=target_name, target_options=["Type 1", "Type 2"],
                             categorical_feature_options=categorical_feature_options,
                             dataset_name="mushrooms")

    return dataset
