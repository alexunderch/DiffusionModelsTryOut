
from diffusers import UNet2DModel
import torch
import torch.nn as nn
from torchvision import transforms as T

class ClassConditionedUnet(nn.Module):
  def __init__(self, n_channels: int, 
               image_size: int, 
               num_classes: int, 
               class_emb_size: int,
               model_size: str = "small") -> None:
    super().__init__()
    assert model_size in ["small", "large"]
    # The embedding layer will map the class label to a vector of size class_emb_size
    #adding a null-token equal to number of classes
    self.class_emb = nn.Embedding(num_classes+1, class_emb_size)
    self.ood = num_classes
    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=image_size,           # the target image resolution
        in_channels=n_channels + class_emb_size, # Additional input channels for class cond.
        out_channels=n_channels,           # the number of output channels
        layers_per_block=2,       # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64), 
        down_block_types=( 
            "DownBlock2D",        # a regular ResNet downsampling block
            "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D", 
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",          # a regular ResNet upsampling block
          ),
    )

  def forward(self, x: torch.Tensor, t: torch.Tensor, class_labels: torch.Tensor = None) -> torch.Tensor:
    bs, _, w, h = x.shape
    if class_labels is None:
        class_labels = (torch.ones(bs) * self.ood).long().to(x.device)
        
    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels) # Map to embedding dinemsion
    class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) 

    # Feed this to the unet alongside the timestep and return the prediction
    return self.model(net_input, t).sample 
  

def clip_loss(image: torch.Tensor, text_features: torch.Tensor, clip_model: nn.Module) -> torch.Tensor: 
  tfms = T.Compose(
      [
          T.RandomResizedCrop(224),
          T.Normalize(
              mean=(0.48145466, 0.4578275, 0.40821073),
              std=(0.26862954, 0.26130258, 0.27577711),
          ),
      ]
  )
  image_features = clip_model.encode_image(tfms(image))  
  input_normed = torch.nn.functional.normalize(image_features.unsqueeze(1), dim=2)
  embed_normed = torch.nn.functional.normalize(text_features.unsqueeze(0), dim=2)
  dists = (
      input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
  )  # Squared Great Circle Distance
  return dists.mean()
     