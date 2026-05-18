from .adult.load import load_data as load_adult_income
from .wine_quality.load import load_data as load_wine_quality
from .heart_disease.load import load_data as load_heart_disease
from .king_county_housing.load import load_data as load_king_county_housing
from .prima_diabetes.load import load_data as load_prima_diabetes
from .mushrooms.load import load_data as load_mushrooms
from .forest_cover.load import load_data as load_forest_cover
from .breast_cancer.load import load_data as load_breast_cancer
from .cardiotocography.load import load_data as load_cardiotocography

def get_dataset(dataset_name, **kwargs):
    if dataset_name == "adult":
        return load_adult_income(**kwargs)

    if dataset_name == "wine_quality":
        return load_wine_quality(**kwargs)

    if dataset_name == "heart_disease":
        return load_heart_disease(**kwargs)

    if dataset_name == "king_county_housing":
        return load_king_county_housing(**kwargs)

    if dataset_name == "prima_diabetes":
        return load_prima_diabetes(**kwargs)


    if dataset_name == "mushrooms":
        return load_mushrooms(**kwargs)


    if dataset_name == "forest_cover":
        return load_forest_cover(**kwargs)


    if dataset_name == "breast_cancer":
        return load_breast_cancer(**kwargs)


    if dataset_name == "cardiotocography":
        return load_cardiotocography(**kwargs)