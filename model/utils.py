import os
from typing import Union

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
from safetensors import safe_open
from tqdm import tqdm

from model.convert_from_ckpt import (
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
)
# from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora, convert_motion_lora_ckpt_to_diffusers
