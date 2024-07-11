"""
Alternative checkpoint for the EigenPlaces model (global descriptor) finetuned for indoor localization.

Model Repo: https://github.com/Enrico-Chiavassa/Indoor-VPR
Orig Paper: EigenPlaces paper (ICCV 2023): https://arxiv.org/abs/2308.10832
"""

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class GeoLocalizationViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "serizba/salad", "dinov2_salad", backbone="dinov2_vitb14"
        )

    def forward(self, images):
        b, c, h, w = images.shape
        # DINO wants height and width as multiple of 14, therefore resize them
        # to the nearest multiple of 14
        h = round(h / 14) * 14
        w = round(w / 14) * 14
        images = tvf.functional.resize(images, [h, w], antialias=True)
        return self.model(images)


class SaladIndoor(BaseModel):
    default_conf = {
        "variant": "salad",
        "backbone": "Dinov2",
        "fc_output_dim": 8448,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        fetched_model = GeoLocalizationViT()

        file_name = f"{conf['variant']}_{conf['backbone']}_{conf['fc_output_dim']}"
        print(f"Loaded model: {file_name}")

        self.net = fetched_model.eval()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data):
        image = self.norm_rgb(data["image"])
        desc = self.net(image)
        return {
            "global_descriptor": desc,
        }
