from os import makedirs
from os.path import join
from pandas import read_csv, DataFrame
from click import command, option
from pickle import load


def get_csv(dir_path: str, table_name: str) -> DataFrame:
    return read_csv(join(dir_path, table_name))

def load_model(dir_path: str, model_name: str):
    with open(join(dir_path, model_name), 'rb') as f:
        return load(f)

@command("predict")
@option("--data-dir")
@option("--models-dir")
@option("--output-dir")
@option("--preprocessor-name")
@option("--model-name")
def predict(data_dir: str, models_dir: str, output_dir: str, preprocessor_name: str, model_name: str):
    data = get_csv(data_dir, "data.csv")

    scaler = load_model(models_dir, preprocessor_name)
    model = load_model(models_dir, model_name)

    data_scaled = scaler.transform(data)
    data["predictions"] = model.predict(data_scaled)

    makedirs(output_dir, exist_ok=True)
    data.to_csv(join(output_dir, "data_pred.csv"))


if __name__ == '__main__':
    predict()