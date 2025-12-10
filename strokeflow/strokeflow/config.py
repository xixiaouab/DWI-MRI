import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional

@dataclass
class StrokeFlowConfig:
    """Global configuration for StrokeFlow."""

    # -------------------------------------------------------------------------
    # 1. Dataset & Preprocessing
    # -------------------------------------------------------------------------
    data_root: str = "./data/ISLES2022"
    img_size: Tuple[int, int, int] = (128, 128, 128)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    num_classes: int = 1

    # -------------------------------------------------------------------------
    # 2. Model Architecture
    # -------------------------------------------------------------------------
    in_channels: int = 2
    base_filters: int = 64
    flow_channels: int = 3
    encoder_name: str = "resnet34"
    pretrained: bool = False

    # -------------------------------------------------------------------------
    # 3. Physics-Informed Loss Function [cite: 497-508]
    # -------------------------------------------------------------------------
    # L_total = lambda_d * L_density + lambda_f * L_flow
    lambda_d: float = 1.0
    lambda_f: float = 0.5

    # L_flow = alpha * L_align + beta * L_smooth + gamma * L_div
    alpha: float = 1.0
    beta: float = 0.2
    gamma: float = 0.1
    
    lesion_threshold: float = 0.1

    # -------------------------------------------------------------------------
    # 4. Optimization
    # -------------------------------------------------------------------------
    optimizer: str = "AdamW"
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    momentum: float = 0.9
    
    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    epochs: int = 200
    warmup_epochs: int = 5
    batch_size: int = 2

    accum_iter: int = 4

    num_workers: int = 4
    seed: int = 42
    amp: bool = True

    # -------------------------------------------------------------------------
    # 6. Logging & Checkpoints
    # -------------------------------------------------------------------------
    experiment_name: str = "StrokeFlow_Baseline"
    log_dir: str = "./runs"
    save_interval: int = 10
    eval_interval: int = 1

    def to_dict(self):
        """Convert config to dict for logging."""
        return asdict(self)

    def save_yaml(self, path: str):
        """Save config to a YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        print(f"Config saved to {path}")

    @classmethod
    def load_yaml(cls, path: str):
        """Load config from a YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
    
        valid_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def __post_init__(self):
      
        assert self.img_size[0] % 32 == 0, "Image size usually should be divisible by 32 (UNet depth 5)"
        assert 0 <= self.lesion_threshold <= 1, "Threshold must be in [0, 1]"


if __name__ == "__main__":
    cfg = StrokeFlowConfig()
    
    cfg.batch_size = 4
    cfg.experiment_name = "Test_Run_v1"
    
    print(cfg)
    
    cfg.save_yaml("./runs/config_test.yaml")
    
    cfg_loaded = StrokeFlowConfig.load_yaml("./runs/config_test.yaml")
    print("Loaded successfully:", cfg_loaded.experiment_name)