from ucimlrepo import fetch_ucirepo 
import os  
from sklearn import preprocessing
from datasets.tabular_dataset import TabularDataset
import pickle


SAVE_PATH = "saved_dataset.pkl"
script_dir = os.path.dirname(__file__)
save_path = os.path.join(script_dir, "saved_dataset.pkl")

def load_data(**kwargs):

    load_previous = kwargs.get("load_previous", False)

    if os.path.exists(save_path) and load_previous:
        with open(save_path, 'rb') as file:
            return pickle.load(file)

    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 

    # data (as pandas dataframes) 
    X = heart_disease.data.features
    y = heart_disease.data.targets 


    
      
    # map_feature_names = {
    #     "trestbps": "resting blood pressure",
    #     "chol": "cholesterol",
    #     "thalach": "max heart rate",
    #     "oldpeak": "ST depression",
    #     "ca": "Number of Colored Vessels",
    #     "thal": "Stress Test Result",
    #     "cp": "Chest Pain Type",
    #     "slope": "slope of ST segment"
    # }

    # sex_map = {0: "female", 1: "male"}
    # thal_map = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}
    # cp_map = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic"}
    # slope_map = {1: "Upslope", 2: "Flat", 3: "Downslope"}

    map_feature_names = {
        "trestbps": "Resting BP",
        "chol": "Cholesterol",
        "thalach": "Max HR",
        "oldpeak": "ST Depress",
        "ca": "Colored Vessels",
        "thal": "Stress Test",
        "cp": "Chest Pain",
        "slope": "ST Slope",
        "sex": "Sex"
    }

    sex_map = {0: "F", 1: "M"}

    thal_map = {
        3: "Norm",
        6: "Fixed",
        7: "Reversible"
    }

    cp_map = {
        1: "Typ. Angina",
        2: "Atyp. Angina",
        3: "Non-Anginal",
        4: "Asymp."
    }

    slope_map = {
        1: "Upslope",
        2: "Flat",
        3: "Downslope"
    }


    # Rename feature columns based on the mapping
    X.rename(columns=map_feature_names, inplace=True)

    feature_names = [col for col in X.columns]

    # Apply mappings to categorical columns
    if 'Sex' in X.columns:
        X['Sex'] = X['Sex'].map(sex_map)
    if 'Stress Test' in X.columns:
        X['Stress Test'] = X['Stress Test'].map(thal_map)
    if 'Chest Pain' in X.columns:
        X['Chest Pain'] = X['Chest Pain'].map(cp_map)
    if 'ST Slope' in X.columns:
        X['ST Slope'] = X['ST Slope'].map(slope_map)

    # Drop missing values
    X = X.dropna()

    # Convert DataFrame to numpy arrays if necessary
    X_numpy = X.values
    y_numpy = ((y.loc[X.index].values)>0)[:, 0]



    # Converts the data from strings into numerical data by counting all the labels
    categorical_feature_options = {}
    categorical_features = [1, 2, 5, 6, 8, 10, 12]
    for feature in categorical_features:
        le = preprocessing.LabelEncoder()
        le.fit(X_numpy[:, feature])
        X_numpy[:, feature] = le.transform(X_numpy[:, feature])
        categorical_feature_options[feature] = le.classes_
    X_numpy = X_numpy.astype(float)


    print(X_numpy.shape)

    # Similar to the X above, we can also preprocess the y to be a numerical value rather than a string
    le = preprocessing.LabelEncoder()
    le.fit(y_numpy)
    y_numerical = le.transform(y_numpy)
    target_options = list(le.classes_)
    target_name = "Heart Condition"


    dataset = TabularDataset(X_numpy, y_numerical, feature_names=feature_names, target_name="condition",
        target_options=["No Heart Condition", "Has Heart Condition"], categorical_feature_options=categorical_feature_options, dataset_name="heart_disease")

    return dataset