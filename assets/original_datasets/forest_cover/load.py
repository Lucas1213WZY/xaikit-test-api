import pandas as pd
import numpy as np
from sklearn import preprocessing
from datasets.tabular_dataset import TabularDataset
import gzip
import os


def load_data(**kwargs):

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "covtype.data.gz")

    # Load the dataset from a local .gz file
    column_names = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", 
                    "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", 
                    "Horizontal_Distance_To_Fire_Points"] + \
                   ["Wilderness_Area_" + str(i) for i in range(4)] + \
                   ["Soil_Type_" + str(i) for i in range(40)] + ["Cover_Type"]
    
    # Open and read the .gz file
    with gzip.open(file_path, 'rt') as f:
        data = pd.read_csv(f, header=None, names=column_names)

    # Select two cover types: Lodgepole Pine (3) and Ponderosa Pine (2)
    cover_types = [3, 4]
    binary_data = data[data['Cover_Type'].isin(cover_types)]

    # Re-label the target: 0 for Ponderosa Pine (2), 1 for Lodgepole Pine (3)
    binary_data['Cover_Type'] = binary_data['Cover_Type'].apply(lambda x: 0 if x == 3 else 1)
    target_name = "Cover_Type"

    # Separate features and target
    X = binary_data.drop(columns=[target_name])
    y = binary_data[target_name]

    feature_names = X.columns.tolist()

    # Identify categorical features (Wilderness_Area and Soil_Type columns)
    categorical_features = ["Wilderness_Area_" + str(i) for i in range(4)] + ["Soil_Type_" + str(i) for i in range(40)]
    categorical_features = [X.columns.get_loc(feature) for feature in categorical_features]

    categorical_feature_options = {}
    X_numpy = X.values

    # Convert categorical features to numerical values
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

    # Convert y to numpy array
    y_numpy = y.values

    # Create the tabular dataset
    dataset = TabularDataset(X_numpy, y_numpy, feature_names=feature_names,
                             target_name=target_name, target_options=["Spruce/Fir", "Lodgepole Pine"], 
                             categorical_feature_options=categorical_feature_options,
                             dataset_name="forest_cover")

    return dataset
