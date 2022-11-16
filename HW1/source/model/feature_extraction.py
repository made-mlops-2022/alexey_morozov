from pandas import DataFrame, read_csv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from numpy import ndarray
from typing import Optional, Tuple
from source.params.dataset_params import DataParams


def get_x_y_from_dataframe(
    dataframe: DataFrame, target_column: str = None
) -> Tuple[DataFrame, Optional[ndarray]]:
    if target_column in dataframe.columns:
        X, y = dataframe.drop([target_column], axis=1), dataframe[target_column].values
    else:
        X, y = dataframe, None
    return X, y


def get_dataframe(data_path):
    return read_csv(data_path)


def get_preprocessor(data_params: DataParams):
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), data_params.numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                data_params.categorical_features,
            ),
        ],
        n_jobs=-1,
    )

    return preprocessor
