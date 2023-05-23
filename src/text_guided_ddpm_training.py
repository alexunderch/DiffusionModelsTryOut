import open_clip
from tqdm.auto import tqdm
from diffusers import UNet2DConditionModel
from metrics import compute_metrics, plot_grid
from omegaconf import DictConfig, OmegaConf
import torch, hydra
import wandb
from hydra.utils import instantiate
from utils import Dataset, NoiseScheduler
from torchmetrics.multimodal.clip_score import CLIPScore

def process_text_with_clip(device: str, clip_model: str):


@hydra.main(version_base=None, config_path="../configs/text_conditioned", config_name="bconfig")
def run(config: DictConfig) -> None:
    wandb.init(config=OmegaConf.to_container(config, resolve=True), 
               project=config.wandb.project_name+f"_ds_{config.dataset.name}", 
               name=config.wandb.name+f"_sch{config.noise_scheduler.name}_model{config.model.model_size}_gr{config.guidance_rate}")
    
    prompt = config.prompt
    guidance_rate = config.guidance_rate
    device = config.device
    train_dataloader, n_channels, image_size, num_classes = instantiate(config.dataset)(config.batch_size, config.num_workers)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(config.clip_model_name, pretrained="openai")
    clip_model.to(device)
    metrics = CLIPScore('openai/clip-vit-base-patch16')

    if config.pipeline is not None:
        pass



if __name__ == "__main__":
    run()