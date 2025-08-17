import os
import string
from dataclasses import dataclass

@dataclass
class Config:
    data_root: str = os.getenv("DATA_ROOT","Dataset_test\captchas")

    chars: str = string.ascii_letters + string.digits
    CAPTCHA_LEN_LOWER_LIMIT: int = 5
    CAPTCHA_LEN_UPPER_LIMIT: int = 7

    RESULT_DIR: str = "Results"
    # Image dimensions - increased for better character detail
    H: int = 60  # Increased from 48 for more vertical detail
    W_max: int = 256  # Increased from 224 for more time steps (T=64)
    grayscale: bool = True
    
    # Model architecture
    total_stride: int = 4  # CNN width downsampling factor
    
    # Training hyperparameters
    batch_size: int = 32  # Local testing
    batch_size_t4: int = 128  # Colab T4 recommendation
    num_workers: int = 4
    amp: bool = True
    
    # Learning rate and optimization
    lr: float = 3e-4
    weight_decay: float = 1e-4
    
    # Training duration
    epochs: int = 40  # For 100k dataset
    epochs_test: int = 10  # For 1k test dataset

cfg = Config()