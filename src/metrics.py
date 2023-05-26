from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torch import Tensor, device as _device, uint8
from typing import Tuple

def compute_fid(generated:Tensor, real:Tensor, device: _device) -> float:
    """Calculates Fréchet inception distance (FID) which is used to access the quality of generated images
        FID = |mu_r - mu_f| + tr(S_r + S_f - 2sqrt(S_r S_f))

    """
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid.update(generated, real=False)
    fid.update(real, real=True)
    return fid.compute().item()

def compute_kid(generated:Tensor, real:Tensor, device: _device, **kwargs) -> Tuple[float, float]:
    """Calculates Kernel Inception Distance (KID) which is used to access the quality of generated images
        KID = MMD(real, fake)^2
    """
    kid = KernelInceptionDistance(feature=2048, 
                                 normalize=True,
                                 subset_size = len(generated) //2,
                                 **kwargs).to(device)
    kid.update(generated, real=False)
    kid.update(real, real=True)
    kid_mean, kid_std = kid.compute()
    return {"mean": kid_mean.item(), "std": kid_std.item() }


def compute_clipscore(generated:Tensor, text:str, device: _device, clip_model_name: str = 'openai/clip-vit-base-patch16'):
    generated = ((generated.clip(-1, 1)*0.5 + 0.5)*255).to(uint8)
    score_model = CLIPScore(clip_model_name).to(device)
    return score_model(generated, text).detach().item()

def compute_metrics(generated:Tensor, real:Tensor, device: _device = _device("cpu"), text:str = None, **kwargs) -> dict:
    metrics = {
        "FID": compute_fid(generated, real, device),
        "KID": compute_kid(generated, real, device, **kwargs) 
    }
    if text is not None:
        metrics.update({"CLIP score": compute_clipscore(generated, text, device)})

    return metrics

