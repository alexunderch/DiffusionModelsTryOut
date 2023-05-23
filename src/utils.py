from diffusers import SchedulerMixin, DDIMScheduler
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Tuple
import torchvision

@dataclass
class Dataset:
    name: str
    root: str
    train: bool
    download: bool

    def __call__(self, batch_size: int, num_workers: int) -> Tuple[DataLoader, int, int, int]:
        if self.name == "MNIST":
            dataset = torchvision.datasets.MNIST(self.root, 
                                                 self.train, 
                                                 transform=torchvision.transforms.ToTensor(), 
                                                 download=self.download)
            return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers), 1, 28, 10

@dataclass
class NoiseScheduler:
    name: SchedulerMixin
    inference_timesteps: int
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = 'linear'

    def __call__(self):
        if self.name == "DDIM":
            nsch = DDIMScheduler(num_train_timesteps=self.num_train_timesteps, beta_schedule=self.beta_schedule)
            nsch.set_timesteps(self.inference_timesteps)
            return nsch