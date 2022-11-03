import argparse
import logging.config
from sklearn.pipeline import Pipeline
from source.params.logging_params import LoggingParams
from source.params.get_params import parse_yaml_file, get_config
from source.params.pipe_params import PipeParams
from source.model.model import get_model, save_model
from source.model.feature_extraction import (
    get_dataframe,
    get_x_y_from_dataframe,
    get_preprocessor,
)


def train(params: PipeParams):
    config = parse_yaml_file(params.logging_params.config_path)
    logging.config.dictConfig(config)
    logger = logging.getLogger(LoggingParams.logger)
    logger.info("Starting training")

    dataframe = get_dataframe(params.data_params.data_path)
    X, y = get_x_y_from_dataframe(dataframe, params.data_params.target)
    preprocessor = get_preprocessor(params.data_params)
    model = get_model(params.model_params)

    pipe = Pipeline([("prep", preprocessor), ("cls", model)])
    pipe.fit(X, y)

    save_model(pipe, params.model_params.model_path)
    save_model(pipe["prep"], params.model_params.preprocessor_path)

    logger.info("Training finished")


def main(config_path):
    params = get_config(config_path)
    train(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training classification model")
    parser.add_argument("-config_path", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config_path)
