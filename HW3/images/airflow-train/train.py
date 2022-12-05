from os import makedirs
from os.path import join
from pandas import read_csv, DataFrame
from click import command, option
from pickle import load, dump
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 17


def get_csv(dir_path: str, table_name: str) -> DataFrame:
    return read_csv(join(dir_path, table_name))


@command("train")
@option("--processed-data-path")
@option("--models-dir")
@option("--preprocessor-name")
@option("--model-name")
def train(processed_data_path: str, models_dir: str, preprocessor_name: str, model_name: str):

    train_data = get_csv(processed_data_path, "X_train.csv")
    train_target = get_csv(processed_data_path, "y_train.csv")

    with open(join(models_dir, preprocessor_name), 'rb') as f:
        scaler = load(f)
    scaled_data = scaler.transform(train_data)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(scaled_data, train_target)

    makedirs(models_dir, exist_ok=True)
    with open(join(models_dir, model_name), 'wb') as f:
        dump(model, f)


if __name__ == '__main__':
    train()
