from unittest import TestCase
from pandas import DataFrame
from source.params.dataset_params import DataParams
from source.model import get_dataframe, get_x_y_from_dataframe, get_preprocessor
from tempfile import TemporaryDirectory
from numpy import ndarray
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os


class TestFeatureExtraction(TestCase):
    filename = "temp.csv"
    df = DataFrame(
        {"feature1": [1, 2, 3], "feature2": [0.12, -3, 0.34], "target": [-1, 0, 1]}
    )

    def test_get_dataframe(self):
        with TemporaryDirectory() as d:
            file_path = os.path.join(d, self.filename)
            self.df.to_csv(file_path, index=False)
            dataframe = get_dataframe(file_path)
            self.assertIsInstance(dataframe, DataFrame)
            self.assertEqual(dataframe.shape, self.df.shape)
            self.assertIn("target", dataframe.columns)

    def test_get_x_y_from_dataframe_with_target(self):
        with TemporaryDirectory() as d:
            file_path = os.path.join(d, self.filename)
            self.df.to_csv(file_path, index=False)
            dataframe = get_dataframe(file_path)
            X, y = get_x_y_from_dataframe(dataframe, "target")
            self.assertIsInstance(y, ndarray)
            self.assertIsInstance(X, DataFrame)
            self.assertEqual(X.shape, (3, 2))

    def test_get_x_y_from_dataframe_without_target(self):
        with TemporaryDirectory() as d:
            file_path = os.path.join(d, self.filename)
            self.df.drop(["target"], axis=1).to_csv(file_path, index=False)
            dataframe = get_dataframe(file_path)
            X, y = get_x_y_from_dataframe(dataframe)
            self.assertIsNone(y)
            self.assertIsInstance(X, DataFrame)
            self.assertEqual(X.shape, (3, 2))

    def test_get_preprocessor(self):
        data_params = DataParams(
            categorical_features=["num1", "num2"],
            numeric_features=["cat1", "cat2"],
            data_path="",
            val_data_path="",
            target="",
        )
        preprocessor = get_preprocessor(data_params)
        self.assertIsInstance(preprocessor, ColumnTransformer)
        self.assertEqual(len(preprocessor.get_params()["transformers"]), 2)
        self.assertIsInstance(preprocessor.get_params()["num"], StandardScaler)
        self.assertIsInstance(preprocessor.get_params()["cat"], OneHotEncoder)
