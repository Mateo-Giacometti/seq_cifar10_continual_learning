from .co2l import train_co2l
from .er_ace import train_er_ace
from .ewc import train_ewc
from .finetuning import train_naive_finetuning
from .lwf import train_lwf

__all__ = [
    "train_co2l",
    "train_er_ace",
    "train_naive_finetuning",
    "train_ewc",
    "train_lwf",
]
