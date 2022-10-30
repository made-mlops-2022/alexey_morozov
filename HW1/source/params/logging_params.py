from dataclasses import dataclass, field

@dataclass
class LoggingParams:
    config_path: str = field(default='configs/logging.yaml')
    logger: str = field(default='main')