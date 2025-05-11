# Slightly modified version of below credit for HQ-SAM
# Credit: https://huggingface.co/Acly/MobileSAM/blob/main/mobile_sam_encoder_onnx/onnx_image_encoder.py

import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Tuple, List

import segment_anything
from segment_anything.modeling import Sam
from segment_anything.utils.amg import calculate_stability_score


class ImageEncoderOnnxModel(nn.Module):
    """
    This model should not be called directly, but is used in ONNX export.
    It combines the image encoder of Sam, with some functions modified to enable
    model tracing. Also supports extra options controlling what information. See
    the ONNX export script for details.
    """

    def __init__(
        self,
        model: Sam,
        # use_tiny_vit: bool,
        use_preprocess: bool,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super().__init__()
        # self.use_tiny_vit = use_tiny_vit
        self.use_preprocess = use_preprocess
        self.pixel_mean = torch.tensor(pixel_mean, dtype=torch.float)
        self.pixel_std = torch.tensor(pixel_std, dtype=torch.float)
        self.image_encoder = model.image_encoder

    @torch.no_grad()
    def forward(self, input_image: torch.Tensor):
        if self.use_preprocess:
            input_image = self.preprocess(input_image)

        # if self.use_tiny_vit:
        #     image_embeddings, interm_embeddings = self.image_encoder(input_image)
        #     return image_embeddings, interm_embeddings
        # else:
        #     img_embeddings, interm_embeddings = self.image_encoder(input_image)
        #     return img_embeddings, torch.stack(interm_embeddings, 0)

        img_embeddings, interm_embeddings = self.image_encoder(input_image)
        return img_embeddings, torch.stack(interm_embeddings, 0)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # permute channels
        x = torch.permute(x, (2, 0, 1))

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))

        # expand channels
        x = torch.unsqueeze(x, 0)
        return x
