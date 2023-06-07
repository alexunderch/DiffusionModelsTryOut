from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Tuple, Callable, Union
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from PIL import Image
import torch

from diffusers import (SchedulerMixin, 
                       DDIMScheduler, 
                       DDPMScheduler, 
                       PNDMScheduler)


@dataclass
class Dataset:
    name: str
    root: str
    train_split: Union[bool, str]
    download: bool

    def _set_image_transform(self, transform: Callable = None) -> None:
        if transform is None:
            self.transform =  torchvision.transforms.Compose([
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                                                 [0.229, 0.224, 0.225])
            ])
        else: 
            self.transform = transform

    def __call__(self, batch_size: int, num_workers: int, img_transform = None) -> Tuple[DataLoader, int, int, int]:
        """
        Returns: dataloader, n_channels, image_size, num_classes
        """
        self._set_image_transform(img_transform)
        def collate_fn(batch):
            return (
                torch.stack([self.transform(x[0]) for x in batch]),
                torch.tensor([x[1] for x in batch])
            )
        if self.name == "MNIST":
            dataset = torchvision.datasets.MNIST(self.root, 
                                                 self.train_split, 
                                                 transform=torchvision.transforms.ToTensor(), 
                                                 download=self.download)
            return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers), 1, 28, 10
        if self.name == "FMNIST":
            dataset = torchvision.datasets.FashionMNIST(self.root, 
                                                        self.train_split, 
                                                        transform=torchvision.transforms.ToTensor(), 
                                                        download=self.download)
            return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers), 1, 28, 10
        if self.name == "EMNIST":
            dataset = torchvision.datasets.EMNIST(self.root, 
                                                        self.train_split, 
                                                        transform=torchvision.transforms.ToTensor(), 
                                                        download=self.download)
            return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers), 1, 28, 10
        if self.name == "LFW":
            dataset = torchvision.datasets.LFWPeople(self.root, 
                                                     split=self.train_split,
                                                    transform=self.transform, 
                                                    download=self.download)
            return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers), 3, 128, 5749
        if self.name == "Flowers":
            dataset = torchvision.datasets.Flowers102(self.root, 
                                                     split=self.train_split,
                                                    download=self.download)
            return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn), 3, 128, 102


@dataclass
class NoiseScheduler:
    name: SchedulerMixin
    inference_timesteps: int
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = 'linear'
    dynamic_thresholding: bool = False

    def __call__(self):
        if self.name == "DDIM":
            nsch = DDIMScheduler
        elif self.name == "DDPM":
            nsch = DDPMScheduler
        elif self.name == "PNDM":
            nsch = PNDMScheduler
            
        nsch = nsch(
            num_train_timesteps=self.num_train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            # thresholding=self.dynamic_thresholding 
        )
        nsch.set_timesteps(self.inference_timesteps)
        return nsch
        
def plot_grid(generated:Tensor, nrow: int) -> Image:
    grid = make_grid(generated, nrow=nrow)
    im = grid.permute(1, 2, 0).cpu().clip(-1, 1)*0.5 + 0.5
    return Image.fromarray(np.array(im*255).astype(np.uint8))

def plot_schedule(scheduler: SchedulerMixin) -> plt.figure:
    fig = plt.figure(figsize=(8, 6))
    plt.plot(scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
    plt.plot((1 - scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
    plt.legend()
    plt.xlabel('timestep'); plt.ylabel(r'$\beta$-schedule value')
    return fig

def set_seed(seed: int = 42) -> None:
    import torch, os
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    return True
