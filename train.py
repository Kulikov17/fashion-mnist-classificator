import os
from pathlib import Path

import hydra
import mlflow
import torch
from configs.config import FashionConfig
from hydra.core.config_store import ConfigStore
from mlflow.utils.git_utils import get_git_branch, get_git_commit
from torch import nn

from fashion.classifier import FashionCNN
from fashion.fitter import Fitter


cs = ConfigStore.instance()
cs.store(name="fashion_config", node=FashionConfig)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: FashionConfig) -> None:
    torch.manual_seed(cfg.params.seed)

    use_cuda = cfg.can_use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    os.system("dvc pull")
    print("START TRAIN, please wait...")

    model = FashionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.lr)
    fitter = Fitter(
        model,
        optimizer,
        criterion,
        device,
        cfg.params.epoch_num,
        cfg.params.batch_size,
        cfg.mlflow_params.logging,
    )

    if cfg.mlflow_params.logging:
        experiment = cfg.mlflow_params.experiment
        experiment_id = mlflow.create_experiment(experiment)
        mlflow.set_experiment(experiment)

        with mlflow.start_run(
            experiment_id=experiment_id, run_name=cfg.mlflow_params.run_name
        ):
            fitter.fit(cfg.files.dataset_dir)
            mlflow.log_param("git commit id", get_git_commit(Path.cwd()))
            mlflow.log_param("git branch", get_git_branch(Path.cwd()))
            mlflow.log_params(dict(cfg.params))
    else:
        fitter.fit(cfg.files.dataset_dir, plot=True)

    os.makedirs(cfg.files.models_dir, exist_ok=True)
    model_filename = cfg.files.models_dir + cfg.files.model_filename
    torch.save(fitter.model.state_dict(), model_filename)


if __name__ == "__main__":
    main()
