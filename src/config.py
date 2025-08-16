import os
import string
from dataclasses import dataclass

@dataclass
class Config:
    data_root: str = os.getenv("DATA_ROOT","Dataset_test\captchas")

    chars: str = string.ascii_letters + string.digits

    H: int = 48
    W_max: int = 224
    grayscale: bool = True  

    total_stride: int = 4  #
    batch_size: int = 32
    num_workers: int = 4
    amp: bool = True  

cfg = Config()