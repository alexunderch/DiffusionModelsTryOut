from diffusers import SchedulerMixin
import torch.nn as nn
import torch
from tqdm.auto import tqdm

@torch.no_grad()
def sampling_loop(
        model: nn.Module,
        noise_scheduler: SchedulerMixin,
        initial_conditions: dict) -> torch.Tensor:
    model.train(False)
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
    model.train(True)
    device = next(model.parameters()).device
    x = batch["sample"].to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
    y = batch["label"].to(device)  if "label" in batch else None
    noise = torch.randn_like(x)
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (len(x),), device=x.device).long()
    noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
    # Get the model prediction
    pred = (1 + guidance_rate) * model(noisy_x, timesteps, y) - guidance_rate * model(noisy_x, timesteps, None)
    return nn.functional.mse_loss(pred, noise)
    
def ttrain_step(
        batch: dict,
        model: nn.Module,
        noise_scheduler: SchedulerMixin,
        guidance_rate: float) -> torch.Tensor:
    model.train(True)
    device = next(model.parameters()).device
    x = batch["sample"].to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))

    y = batch["text"]
    noise = torch.randn_like(x)
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (len(x),), device=x.device).long()
    noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
    noise_pred_text, noise_pred_uncond = model(noisy_x, timesteps, text=y).chunk(2)
    # Get the model prediction
    pred = (1 + guidance_rate) * noise_pred_text - guidance_rate *  noise_pred_uncond
    return nn.functional.mse_loss(pred, noise)


@torch.no_grad()
def tsampling_loop(
        model: nn.Module,
        noise_scheduler: SchedulerMixin,
        initial_conditions: dict) -> torch.Tensor:
    model.train(False)
    device = next(model.parameters()).device

    noise_x = initial_conditions["sample"].to(device)
    y = initial_conditions["text"]  
    
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
        model_input = noise_scheduler.scale_model_input(noise_x, t)
        noise_pred, _ = model(model_input, t, text=y).chunk(2)
        noise_x = noise_scheduler.step(noise_pred, t, noise_x).prev_sample
    return noise_x



