"""
Alternative checkpoint for the EigenPlaces model (global descriptor) finetuned for indoor localization.

Model Repo: https://github.com/Enrico-Chiavassa/Indoor-VPR
Orig Paper: EigenPlaces paper (ICCV 2023): https://arxiv.org/abs/2308.10832
"""

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel

AVAILABLE_VARIATIONS = {
    "eigenplaces_ResNet50_2048": ["GB1_BAI_5_10", "GB1_BAI_10_25_S"],
}


class GeoLocalizationNet(torch.nn.Module):
    """The used networks are composed of a backbone and an aggregation layer."""

    def __init__(self, backbone: str, fc_output_dim: int):
        super().__init__()
        self.model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone=backbone,
            fc_output_dim=fc_output_dim,
        )

    def forward(self, x):
        x = self.model(x)
        return x


class EigenPlacesIndoor(BaseModel):
    default_conf = {
        "variant": "eigenplaces",  # "eigenplaces" not "EigenPlaces" for the indoor model
        "backbone": "ResNet50",
        "fc_output_dim": 2048,
        "variation": "GB1_BAI_10_25_S",
    }
    required_inputs = ["image"]

    def _init(self, conf):
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        fetched_model = GeoLocalizationNet(conf["backbone"], conf["fc_output_dim"])

        file_name = f"{conf['variant']}_{conf['backbone']}_{conf['fc_output_dim']}"
        var_name = conf["variation"]
        file_name += f"_{var_name}"

        fetched_model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                f"https://github.com/Enrico-Chiavassa/Indoor-VPR/releases/download/v0.1.0/{file_name}.pth",
            )
        )
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
