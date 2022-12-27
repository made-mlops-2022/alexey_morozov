from os.path import join
from os import makedirs
from click import command, option
from sklearn.datasets import load_digits


@command("download")
@option("--output_path")
def download(output_path: str):
    makedirs(output_path, exist_ok=True)
    X, y = load_digits(return_X_y=True, as_frame=True)
    X.to_csv(join(output_path, 'data.csv'), index=False)
    y.to_csv(join(output_path, 'target.csv'), index=False)


if __name__ == '__main__':
    download()