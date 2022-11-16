import os
from unittest import TestCase
from source.model import get_model, load_model, save_model
from source.params import ModelParams
from tempfile import TemporaryDirectory
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class TestModel(TestCase):
    def test_get_model(self):
        model_params = ModelParams(model="lr", model_path="", preprocessor_path="")
        model = get_model(model_params)
        self.assertIsInstance(model, LogisticRegression)

        model_params = ModelParams(model="rf", model_path="", preprocessor_path="")
        model = get_model(model_params)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_save_load_model(self):
        C = 1e-3
        model = LogisticRegression(C=C)
        model_path = "model.pickle"
        with TemporaryDirectory() as d:
            file_path = os.path.join(d, model_path)
            save_model(model, file_path)
            self.assertTrue(os.path.exists(file_path))

            loaded_model = load_model(file_path)
            self.assertIsInstance(loaded_model, LogisticRegression)
            self.assertEqual(loaded_model.get_params()["C"], C)
