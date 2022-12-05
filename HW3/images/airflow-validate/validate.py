from os import makedirs
from os.path import join
from pandas import read_csv, DataFrame
from click import command, option
from pickle import load
from json import dump
from sklearn.metrics import classification_report


def get_csv(dir_path: str, table_name: str) -> DataFrame:
    return read_csv(join(dir_path, table_name))

@command("validate")
@option("--data-dir")
@option("--models-dir")
@option("--output-dir")
@option("--preprocessor-name")
@option("--model-name")
def validate(data_dir: str, models_dir: str, output_dir: str, preprocessor_name: str, model_name: str):
    val_data = get_csv(data_dir, 'X_val.csv')
    val_target = get_csv(data_dir, 'y_val.csv')

    with open(join(models_dir, preprocessor_name), 'rb') as f:
        scaler = load(f)
    scaled_data = scaler.transform(val_data)

    with open(join(models_dir, model_name), 'rb') as f:
        model = load(f)

    y_pred = model.predict(scaled_data)

    makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, 'classification_report.txt'), 'w') as f:
        dump(classification_report(val_target, y_pred, output_dict=True), f)


if __name__ == '__main__':
    validate()
