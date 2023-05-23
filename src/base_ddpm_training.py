from tqdm.auto import tqdm
from metrics import compute_metrics, plot_grid
from models import ClassConditionedUnet
from scheduling import sampling_loop, train_step
import wandb, hydra
from omegaconf import DictConfig, OmegaConf
import torch
from hydra.utils import instantiate
from utils import Dataset, NoiseScheduler


@hydra.main(version_base=None, config_path="../configs/class_conditioned/", config_name="bconfig")
def run(config: DictConfig) -> None:
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
    opt = torch.optim.AdamW(net.parameters(), lr=config.lr) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config.exp_lr_schedule)

    for epoch in range(config.n_epochs):
        for step, (x, y) in  tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            loss = train_step(batch={"sample": x, "label": y}, 
                              model=net, 
                              noise_scheduler=noise_scheduler, 
                              guidance_rate=guidance_rate)
            loss.backward()
            wandb.log({"loss": loss.item()})

            if (step+1)%config.grad_accumulation_steps==0:
                opt.zero_grad()
                opt.step()
                
            if (step+1)%config.log_samples_every==0:
                noise_x = torch.randn(10, n_channels, image_size, image_size).to(device) # Batch of 10
                y = torch.arange(0, 10).to(device)
                real = x
                generated = sampling_loop(net, noise_scheduler, {"sample": noise_x, "label": y}) 
                wandb.log(compute_metrics(generated.expand(-1, 3,-1,-1), real.expand(-1, 3,-1,-1)))
                wandb.log({f'Sample generations': wandb.Image(plot_grid(generated, nrow=num_classes//2))})
        scheduler.step()



if __name__ == "__main__":
    run()