
from diffusers import UNet2DModel, UNet2DConditionModel
import torch
import torch.nn as nn
from torchvision import transforms as T
from typing import Callable
from transformers import AutoTokenizer, CLIPTextModel, logging
logging.set_verbosity_error()

def predifined_model_config(model_size: str = None):
    config = {}
    if model_size == "small":
        config = dict(layers_per_block=2,       # how many ResNet layers to use per UNet block
                    block_out_channels=(32, 64, 64), 
                    down_block_types=( 
                        "DownBlock2D",        # a regular ResNet downsampling block
                        "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                        "AttnDownBlock2D"), 
                    up_block_types=(
                        "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                        "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                        "UpBlock2D")
                    )
    if model_size == "large":
        config = dict(layers_per_block=1,  # how many ResNet layers to use per UNet block
                    block_out_channels=(128, 128, 256, 256, 512),  # the number of output channels for each UNet block
                    down_block_types=(
                                        "DownBlock2D",
                                        "DownBlock2D",
                                        "DownBlock2D",
                                        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                                        "DownBlock2D"),
                    up_block_types=(
                        "UpBlock2D",  # a regular ResNet upsampling block
                        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                        "UpBlock2D",
                        "UpBlock2D",
                        "UpBlock2D")
                    )
    return config

class ClassConditionedUnet(nn.Module):
    def __init__(self, n_channels: int, 
                       image_size: int, 
                       num_classes: int, 
                       class_emb_size: int,
                       model_size: str = "small") -> None:
        super().__init__()
        assert model_size in ["small", "large", None]
        # The embedding layer will map the class label to a vector of size class_emb_size
        #adding a null-token equal to number of classes
        self.class_emb = nn.Embedding(num_classes+1, class_emb_size)
        self.ood = num_classes
        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=image_size,           # the target image resolution
            in_channels=n_channels + class_emb_size, # Additional input channels for class cond.
            out_channels=n_channels,           # the number of output channels
            **predifined_model_config(model_size)
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
        return self.model(net_input, t, return_dict=False)[0]
    
class TextConditionedUnet(nn.Module):
    def __init__(self, n_channels: int,  
               image_size: int, 
               clip_model: Callable = None) -> None:
        super().__init__()


        self.tmodel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.model = UNet2DConditionModel(
            sample_size=image_size,    
            in_channels=n_channels,
            out_channels=n_channels,          
            layers_per_block=1,
            block_out_channels=(32, 64, 64), 
            down_block_types = (
                'CrossAttnDownBlock2D',
                'CrossAttnDownBlock2D', 
                'DownBlock2D'
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",          # a regular ResNet upsampling block
                ),
            cross_attention_dim=512,
            addition_embed_type="text")   
              
        self.encoder_hid_proj = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, text: str = None) -> torch.Tensor:
        bs, _, w, h = x.shape

        x = torch.cat((x, x))
        t = torch.cat((t, t)) if t.numel() > 1 else t.item() * torch.ones((2 * bs, )).long().to(x.device)


        with torch.no_grad():
            text_embeddings_c = self.tmodel(**{k: v.to(x.device) for k, v in self.tokenizer([text] * bs, 
                                                                                            padding=True, 
                                                                                            return_tensors="pt").items()}).last_hidden_state  
            text_embeddings_u = self.tmodel(**{k: v.to(x.device) for k, v in self.tokenizer([""] * bs, 
                                                                                            padding="max_length", 
                                                                                            max_length = text_embeddings_c.size(1), 
                                                                                            return_tensors="pt").items()}).last_hidden_state
        hs = torch.cat((self.encoder_hid_proj(text_embeddings_c), 
                        self.encoder_hid_proj(text_embeddings_u))) 
        return self.model(x, t, encoder_hidden_states=hs).sample

