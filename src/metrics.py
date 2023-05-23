from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid
import numpy as np
from torch import Tensor
from PIL import Image
from typing import Tuple

def compute_fid(generated:Tensor, real:Tensor) -> float:
    """Calculates Fréchet inception distance (FID) which is used to access the quality of generated images
        FID = |mu_r - mu_f| + tr(S_r + S_f - 2sqrt(S_r S_f))

    """
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    fid.update(generated, real=False)
    fid.update(real, real=True)
    return fid.compute().item()

def compute_kid(generated:Tensor, real:Tensor, **kwargs) -> Tuple[float, float]:
    """Calculates Kernel Inception Distance (KID) which is used to access the quality of generated images
        KID = MMD(real, fake)^2
    """
    kid = KernelInceptionDistance(feature=2048, 
                                 normalize=True,
                                 subset_size = len(generated) //2,
                                 **kwargs)
    kid.update(generated, real=False)
    kid.update(real, real=True)
    kid_mean, kid_std = kid.compute()
    return {"mean": kid_mean.item(), "std": kid_std.item() }

def compute_metrics(generated:Tensor, real:Tensor, **kwargs) -> dict:
    return {
        "FID": compute_fid(generated, real),
        "KID": compute_kid(generated, real, **kwargs) 
    }


def plot_grid(generated:Tensor, nrow: int) -> Image:
    grid = make_grid(generated, nrow=nrow)
    im = grid.permute(1, 2, 0).cpu().clip(-1, 1)*0.5 + 0.5
    return Image.fromarray(np.array(im*255).astype(np.uint8))