import argparse
import logging.config
from typing import Any
from sklearn.metrics import classification_report
from source.params.logging_params import LoggingParams
from source.params.get_params import parse_yaml_file, get_config
from source.model.model import load_model
from source.params.pipe_params import PipeParams
from source.model.feature_extraction import (
    get_dataframe,
    get_x_y_from_dataframe
)


def evaluate_model(params: PipeParams) -> dict[str, Any]:
    config = parse_yaml_file(params.logging_params.config_path)
    logging.config.dictConfig(config)
    logger = logging.getLogger(LoggingParams.logger)
    logger.info("Evaluating...")

    dataframe = get_dataframe(params.data_params.val_data_path)
    X, y = get_x_y_from_dataframe(dataframe, params.data_params.target)

    model = load_model(params.model_params.model_path)

    predictions = model.predict(X)
    report = classification_report(y, predictions, output_dict=True)
    logger.debug("\n" + classification_report(y, predictions))

    return report


def main(config_path):
    params = get_config(config_path)
    report = evaluate_model(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating classification model")
    parser.add_argument("-config_path", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config_path)
