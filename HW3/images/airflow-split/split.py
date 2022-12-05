from os import makedirs
from os.path import join
from numpy import ndarray
from pandas import read_csv, DataFrame
from click import command, option
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import dump

VAL_SIZE = 0.25
RANDOM_STATE = 17


def get_csv(dir_path: str, table_name: str) -> DataFrame:
    return read_csv(join(dir_path, table_name))

def to_csv(array_data: ndarray, dir_path: str, table_name: str):
    DataFrame(array_data).to_csv(join(dir_path, table_name), index=False)

@command("split")
@option("--processed-data-path")
@option("--output-dir")
@option("--models-dir")
@option("--preprocessor-name")
def split(processed_data_path: str, output_dir: str, models_dir: str, preprocessor_name: str):
    train_data = get_csv(processed_data_path, "train_data.csv")
    X_train, X_val, y_train, y_val = train_test_split(train_data.drop(['target'], axis=1), train_data.target, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    
    makedirs(output_dir, exist_ok=True)
    to_csv(X_train, output_dir, 'X_train.csv')
    to_csv(X_val, output_dir, 'X_val.csv')
    to_csv(y_train, output_dir, 'y_train.csv')
    to_csv(y_val, output_dir, 'y_val.csv')
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)

    to_csv(X_train_scaled, output_dir, 'X_train_scaled')
    to_csv(X_val_scaled, output_dir, 'X_val_scaled')

    makedirs(models_dir, exist_ok=True)
    with open(join(models_dir, preprocessor_name), 'wb') as f:
        dump(sc, f)



if __name__ == '__main__':
    split()
