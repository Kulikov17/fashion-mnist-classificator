import hydra
import torch
from config import FashionConfig
from hydra.core.config_store import ConfigStore

from fashion.classifier import FashionCNN
from fashion.inference import inference


cs = ConfigStore.instance()
cs.store(name="fashion_config", node=FashionConfig)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: FashionConfig) -> None:
    torch.manual_seed(cfg.params.seed)

    use_cude = cfg.can_use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cude else "cpu")

    model = FashionCNN()
    model_filename = cfg.files.models_dir + cfg.files.model_filename
    model.load_state_dict(torch.load(model_filename))

    y_true, y_pred = inference(
        cfg.files.dataset_dir, model, device, cfg.params.batch_size
    )

    print(y_true, y_pred)


if __name__ == "__main__":
    main()
