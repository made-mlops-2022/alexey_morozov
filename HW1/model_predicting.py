import argparse
import logging.config
from pandas import DataFrame
from source.params.logging_params import LoggingParams
from source.params.get_params import parse_yaml_file, get_config
from source.model.model import load_model
from source.params.pipe_params import PipeParams


from source.model.feature_extraction import get_dataframe, get_x_y_from_dataframe


def predict(params: PipeParams):
    config = parse_yaml_file(params.logging_params.config_path)
    logging.config.dictConfig(config)
    logger = logging.getLogger(LoggingParams.logger)
    logger.info("Starting predicting")

    dataframe = get_dataframe(params.data_params.data_path)
    X, _ = get_x_y_from_dataframe(dataframe)
    model = load_model(params.model_params.model_path)

    predictions = model.predict(X)
    DataFrame(predictions, columns=["predicted_label"], index=dataframe.index).to_csv(
        params.pred_path
    )
    logger.info("Predictions saved: " + params.pred_path)


def main(config_path):
    params = get_config(config_path)
    predict(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Getting predictions")
    parser.add_argument("-config_path", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config_path)
