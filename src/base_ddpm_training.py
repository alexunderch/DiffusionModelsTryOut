from tqdm.auto import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from metrics import compute_metrics
from models import ClassConditionedUnet
from scheduling import sampling_loop, train_step
import wandb, hydra
from omegaconf import DictConfig, OmegaConf
import torch
from hydra.utils import instantiate
from utils import (Dataset, 
                   NoiseScheduler,
                   plot_schedule, 
                   plot_grid, 
                   set_seed)


@hydra.main(version_base=None, config_path="../configs/class_conditioned/", config_name="bconfig")
def run(config: DictConfig) -> None:
    set_seed(config.seed)
    # print(OmegaConf.to_yaml(config, resolve=True))
    wandb.init(config=OmegaConf.to_container(config, resolve=True), 
               project=config.wandb.project_name+f"_ds_{config.dataset.name}", 
               name=config.wandb.name+f"_sch{config.noise_scheduler.name}_model{config.model.model_size}_gr{config.guidance_rate}")
    
    guidance_rate = config.guidance_rate
    device = config.device
    train_dataloader, n_channels, image_size, num_classes = instantiate(config.dataset)(config.batch_size, config.num_workers)

    net = ClassConditionedUnet(n_channels, 
                               image_size, 
                               num_classes=num_classes, 
                               class_emb_size=config.model.class_emb_size,
                               model_size=config.model.model_size).to(device)

    noise_scheduler = instantiate(config.noise_scheduler)()
    sampling_noise_scheduler = instantiate(config.noise_scheduler_sample)()

    wandb.log({"Noise scheduler": plot_schedule(noise_scheduler)})

    opt = torch.optim.AdamW(net.parameters(), lr=config.lr) 
    scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config.exp_lr_schedule)
    scheduler = get_cosine_schedule_with_warmup(optimizer=opt,
                                                 num_warmup_steps=config.lr_warmup_steps,
                                                 num_training_steps=(len(train_dataloader) * config.n_epochs))

    for epoch in range(config.n_epochs):
        for step, (x, y) in  tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            loss = train_step(batch={"sample": x, "label": y}, 
                              model=net, 
                              noise_scheduler=noise_scheduler, 
                              guidance_rate=guidance_rate)
            wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
            
            loss.backward()

            if (step+1)%config.grad_accumulation_steps==0:
                opt.zero_grad()

                opt.step()
                scheduler.step()
                
            if (step+1)%config.log_samples_every==0:
                noise_x = torch.randn(num_classes, n_channels, image_size, image_size).to(device) # Batch of 10
                y = torch.arange(0, num_classes).to(device)
                real = x.to(device)
                generated = sampling_loop(net, sampling_noise_scheduler, {"sample": noise_x, "label": y}).to(device) 
                wandb.log(compute_metrics(generated.expand(-1, 3,-1,-1), real.expand(-1, 3,-1,-1), device=device))
                wandb.log({f'Sample generations': wandb.Image(plot_grid(generated, nrow=num_classes//4))})
        scheduler_2.step(epoch)


if __name__ == "__main__":
    run()