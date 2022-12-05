from os import makedirs
from os.path import join
from pandas import read_csv, DataFrame, concat
from click import command, option

def get_csv(data_path: str, table_name: str) -> DataFrame:
    return read_csv(join(data_path, table_name))


@command("predict")
@option("--raw-data-path")
@option("--processed-data-path")
def preprocess(raw_data_path: str, processed_data_path: str):
    X = get_csv(raw_data_path, "data.csv")
    y = get_csv(raw_data_path, "target.csv")
    makedirs(processed_data_path, exist_ok=True)
    concat([X, y], axis=1).to_csv(join(processed_data_path, "train_data.csv"), index=False)


if __name__ == '__main__':
    preprocess()