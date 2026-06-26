import numpy as np
import pickle
import os
from sklearn import preprocessing
from datasets.tabular_dataset import TabularDataset


SAVE_PATH = "saved_dataset.pkl"
script_dir = os.path.dirname(__file__)
save_path = os.path.join(script_dir, "saved_dataset.pkl")


def _normalize_adult_dataset(dataset):
    """Keep cached Adult feature metadata aligned with the 14-column data matrix."""
    feature_names = getattr(dataset, "feature_names", None)
    X = getattr(dataset, "X", None)
    target_name = getattr(dataset, "target_name", None)

    if (
        feature_names
        and X is not None
        and len(feature_names) == X.shape[1] + 1
        and feature_names[-1] == target_name
    ):
        dataset.feature_names = feature_names[:-1]

    return dataset


def load_data(**kwargs):

    load_previous = kwargs.get("load_previous", True)

    if os.path.exists(save_path) and load_previous:
        with open(save_path, 'rb') as file:
            return _normalize_adult_dataset(pickle.load(file))

    try:
        from ucimlrepo import fetch_ucirepo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Adult dataset can be loaded from the local cache by calling "
            "`load_data(load_previous=True)`. To refresh it from UCI, install "
            "`ucimlrepo` first, for example `pip install ucimlrepo`."
        ) from exc

    # fetch dataset 
    adult = fetch_ucirepo(id=2) 
      
    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets

    X = X.dropna()
    y = y.loc[X.index]

    def map_array_values(array, value_map):
        # value map must be { src : target }
        ret = array.copy()
        for src, target in value_map.items():
            ret[ret == src] = target
        return ret

    feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                             "Years of Education", "Marital Status", "Occupation",
                             "Relationship", "Race", "Sex", "Capital Gain",
                             "Capital Loss", "Hours per week", "Country"]
    # # features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # categorical_features = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
    categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
    education_map = {
        '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
        'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
        'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
        'Some-college': 'High School grad', 'Masters': 'Masters',
        'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
        'Assoc-voc': 'Associates',
    }
    occupation_map = {
        "Adm-clerical": "Admin", "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
        "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
        "Service", "Priv-house-serv": "Service", "Prof-specialty":
        "Professional", "Protective-serv": "Other", "Sales":
        "Sales", "Tech-support": "Other", "Transport-moving":
        "Blue-Collar",
    }
    country_map = {
        'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
        'China', 'Columbia': 'South-America', 'Cuba': 'Other',
        'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
        'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
        'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
        'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
        'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
        'Hong': 'China', 'Hungary': 'Euro_2', 'India':
        'British-Commonwealth', 'Iran': 'Other', 'Ireland':
        'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
        'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
        'Latin-America', 'Nicaragua': 'Latin-America',
        'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
        'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
        'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
        'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
        'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
        'United-States': 'United-States', 'Vietnam': 'SE-Asia'
    }
    married_map = {
        'Never-married': 'no', 'Married-AF-spouse': 'yes',
        'Married-civ-spouse': 'yes', 'Married-spouse-absent':
        'separated', 'Separated': 'separated', 'Divorced':
        'separated', 'Widowed': 'widowed'
    }

    def cap_gains_fn(x):
        x = x.astype(float)
        d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                        right=True).astype(str)
        return map_array_values(d, {'0': 'No gains', '1': 'Low gains', '2': 'High gains'})

    def cap_loss_fn(x):
        x = x.astype(float)
        d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                        right=True).astype(str)
        return map_array_values(d, {'0': 'No losses', '1': 'Low losses', '2': 'High losses'})

    transformations = {
        3: lambda x: map_array_values(x, education_map),
        5: lambda x: map_array_values(x, married_map),
        6: lambda x: map_array_values(x, occupation_map),
        # 10: cap_gains_fn,
        # 11: cap_loss_fn,
        13: lambda x: map_array_values(x, country_map),
    }



    X_numpy = X.values
    # Apply all the above transformations
    for feature, fun in transformations.items():
            X_numpy[:, feature] = fun(X_numpy[:, feature])

    y_numpy = y.values[:, 0]
    # Some defect in the dataset where some labels have "." at their end. The below simply removes the "."
    for i, label in enumerate(y_numpy):
        if label[-1]==".":
                y_numpy[i] = label[:-1]


    # Converts the data from strings into numerical data by counting all the labels
    categorical_feature_options = {}
    for feature in categorical_features:
        le = preprocessing.LabelEncoder()
        le.fit(X_numpy[:, feature])
        X_numpy[:, feature] = le.transform(X_numpy[:, feature])
        categorical_feature_options[feature] = le.classes_
    X_numpy = X_numpy.astype(float)

    # Similar to the X above, we can also preprocess the y to be a numerical value rather than a string
    le = preprocessing.LabelEncoder()
    le.fit(y_numpy)
    y_numerical = le.transform(y_numpy)
    target_options = list(le.classes_)
    target_name = "Income"


    dataset = TabularDataset(X_numpy, y_numerical, feature_names=feature_names,\
        categorical_feature_options=categorical_feature_options, target_name=target_name,\
        target_options=["Low Income", "High Income"], dataset_name="adult")
    dataset = _normalize_adult_dataset(dataset)


    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    return dataset

'''
python
import datasets
data = datasets.get_dataset("adult")
train, dev, test = data.use_specific_features(["Age", "Workclass", "fnlwgt", "Capital Gain", "Occupation"]).split()
X, y = train.prepare_features_for_model()
'''
