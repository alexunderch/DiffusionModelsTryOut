import open_clip
from tqdm.auto import tqdm
from models import TextConditionedUnet
from metrics import compute_metrics
from omegaconf import DictConfig, OmegaConf
import torch, hydra
import wandb
from hydra.utils import instantiate
from scheduling import ttrain_step, tsampling_loop
from utils import (Dataset, 
                   NoiseScheduler, 
                   plot_grid, 
                   plot_schedule, 
                   set_seed)



@hydra.main(version_base=None, config_path="../configs/text_conditioned", config_name="bconfig")
def run(config: DictConfig) -> None:
    set_seed(config.seed)

    wandb.init(config=OmegaConf.to_container(config, resolve=True), 
               project=config.wandb.project_name+f"_ds_{config.dataset.name}", 
               name=config.wandb.name+f"_sch{config.noise_scheduler.name}_gr{config.guidance_rate}")
    
    prompt = config.prompt
    guidance_rate = config.guidance_rate
    device = config.device
    train_dataloader, n_channels, image_size, num_classes = instantiate(config.dataset)(config.batch_size, config.num_workers)

    net = TextConditionedUnet(n_channels, 
                               image_size).to(device)

    noise_scheduler = instantiate(config.noise_scheduler)()
    wandb.log({"Noise scheduler": plot_schedule(noise_scheduler)})

    opt = torch.optim.AdamW(net.parameters(), lr=config.lr) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config.exp_lr_schedule)
    sampling_noise_scheduler = instantiate(config.noise_scheduler_sample)()


    for epoch in range(config.n_epochs):
        for step, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            loss = ttrain_step(batch={"sample": x, "text": prompt}, 
                              model=net, 
                              noise_scheduler=noise_scheduler, 
                              guidance_rate=guidance_rate)
            wandb.log({"loss": loss.item()})
            #loss.backward()

            if (step+1)%config.grad_accumulation_steps==0:
                opt.zero_grad()
                loss.backward()
                opt.step()
                
            if (step+1)%config.log_samples_every==0:
                noise_x = torch.randn(4, n_channels, image_size, image_size).to(device) # Batch of 10
                real = x.to(device)
                generated = tsampling_loop(net, sampling_noise_scheduler, {"sample": noise_x, "text": prompt}) 
                # wandb.log(compute_metrics(generated.expand(-1, 3,-1,-1), real.expand(-1, 3,-1,-1), device=device, text=[prompt]*len(generated)))
                wandb.log({f'Sample generations': wandb.Image(plot_grid(generated, nrow=1))})
        scheduler.step(epoch)




if __name__ == "__main__":
    run()
