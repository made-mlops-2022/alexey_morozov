import logging.config
import yaml
from source.params.logging_params import LoggingParams
from source.training.model import get_model
from source.params.pipe_params import PipeParams
from source.params import get_params
from sklearn.pipeline import Pipeline
import argparse

from source.training.feature_extraction import get_dataframe, get_x_y_from_dataframe

def predict(params: PipeParams):
    with open(params.logging_params.config_path, 'r') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logger = logging.getLogger(LoggingParams.logger)
    logger.info('Starting predicting')

    dataframe = get_dataframe(params.data_params.data_path)
    X, _ = get_x_y_from_dataframe(dataframe)
    preprocessor = get_model(params.model_params.preprocessor_path)
    model = get_model(params.model_params.model_path)

    pipe = Pipeline([
        ('prep', preprocessor),
        ('cls', model)
    ])
    pred = pipe.predict(X)
    pred.to_csv(params.pred_path)
    logger.info('Predictions saved:',  params.pred_path)


def main(config_path):
    params = get_params(config_path)
    predict(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Getting predictions")
    parser.add_argument("-config_path", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config_path)