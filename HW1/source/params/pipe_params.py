from source.params.dataset_params import DataParams
from source.params.model_params import ModelParams
from source.params.logging_params import LoggingParams


class PipeParams:
    data_params: DataParams
    logging_params: LoggingParams
    model_params: ModelParams
    pred_path: str
