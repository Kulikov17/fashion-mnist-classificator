from dataclasses import dataclass


@dataclass
class Files:
    dataset_dir: str
    models_dir: str
    model_filename: str
    results_dir: str
    results_filename: str


@dataclass
class Params:
    seed: int
    epoch_num: int
    batch_size: int
    lr: float


@dataclass
class MlflowParams:
    logging: bool
    experiment: str


@dataclass
class FashionConfig:
    can_use_cuda: bool
    files: Files
    params: Params
    mlflow_params: MlflowParams
