# Small Diffusion models project
Heavily based on HuggingFace `Diffusers` library

## Installation
Tested on python 3.9, 3.10

```Bash
pip install -r requirements.txt
```

## Usage
The project uses `wandb` for logging
```Bash
wandb login
```
To setup a base DDPM training experiment:
```Bash
python src/base_ddpm_training.py
```

To setup text-conditioned training with a given prompt
```Bash
python src/base_ddpm_training.py prompt=YOURPROMPT
```

The project uses [hydra](https://hydra.cc/) to power experiments. You can modify experiments according with configs structure 

```
.
├── class_conditioned
│   ├── bconfig.yaml
│   ├── dataset
│   │   ├── emnist.yaml
│   │   ├── fashion_mnist.yaml
│   │   ├── flowers.yaml
│   │   ├── lfw.yaml
│   │   └── mnist.yaml
│   ├── noise_scheduler
│   │   ├── ddim.yaml
│   │   ├── ddpm.yaml
│   │   └── pndm.yaml
│   └── noise_scheduler_sample
│       └── ddim.yaml
└── text_conditioned
    ├── bconfig.yaml
    ├── dataset
    │   └── mnist.yaml
    ├── noise_scheduler
    │   ├── ddim.yaml
    │   ├── ddpm.yaml
    │   └── pndm.yaml
    └── noise_scheduler_sample
        └── ddim.yaml
```

## Basic training config (`bconfig.yaml`)
```
wandb:
  project_name: ddpm_training
  name: str
defaults:
  - dataset: mnist/fashion_mnist/...
  - noise_scheduler: ddpm/ddim/pndm/...
  - noise_scheduler_sample: ddim
model:
  model_size: small/large
  class_emb_size: 6
seed: 42
lr_warmup_steps: 0.2
exp_lr_schedule: 0.95
batch_size: 128
num_workers: 2
guidance_rate: 0
device: cuda/cpu
lr: 1.0e-3
n_epochs: 8
log_samples_every: 100
grad_accumulation_steps: 1
```
