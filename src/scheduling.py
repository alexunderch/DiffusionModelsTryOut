from diffusers import SchedulerMixin
import torch.nn as nn
import torch
from tqdm.auto import tqdm
from models import clip_loss

@torch.no_grad()
def sampling_loop(
        model: nn.Module,
        noise_scheduler: SchedulerMixin,
        initial_conditions: dict) -> torch.Tensor:
    model.train(True)
    noise_x = initial_conditions["sample"]
    y = initial_conditions["label"] if "label" in initial_conditions else None
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
        model_input = noise_scheduler.scale_model_input(noise_x, t)
        noise_pred = model(model_input, t, y)
        noise_x = noise_scheduler.step(noise_pred, t, noise_x).prev_sample
    return noise_x
    
def train_step(
        batch: dict,
        model: nn.Module,
        noise_scheduler: SchedulerMixin,
        guidance_rate: float) -> torch.Tensor:
    device = next(model.parameters()).device
    x = batch["sample"].to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
    y = batch["label"].to(device)  if "label" in batch else None
    noise = torch.randn_like(x)
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (len(x),), device=x.device).long()
    noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
    # Get the model prediction
    pred = (1 + guidance_rate) * model(noisy_x, timesteps, None) - guidance_rate * model(noisy_x, timesteps, y)
    return nn.functional.mse_loss(pred, noise)


@torch.no_grad()
def guidance_step(
        model: nn.Module,
        noise_scheduler: SchedulerMixin,
        initial_conditions: dict) -> torch.Tensor:
    model.train(True)
    noise_x = initial_conditions["sample"]
    y = initial_conditions["label"] if "label" in initial_conditions else None
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
        model_input = noise_scheduler.scale_model_input(noise_x, t)
        noise_pred = model(model_input, t, y)
        noise_x = noise_scheduler.step(noise_pred, t, noise_x).prev_sample
    return noise_x
    


