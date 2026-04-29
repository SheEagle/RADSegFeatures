import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoModel


@dataclass
class SpatialFeatures:
    feature_map: torch.Tensor


class VisionLanguageBackend:
    def __init__(self, device="cuda"):
        self.device = device
        self.transform = None

    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode_image_to_feature_map(self, img_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RADSegBackend(VisionLanguageBackend):
    def __init__(self, model_version="c-radio_v4-h", lang_model="siglip2-g", device="cuda"):
        super().__init__(device=device)
        print(f"Loading RADSeg model {model_version}...")
        self.model = torch.hub.load(
            "RADSeg-OVSS/RADSeg",
            "radseg_encoder",
            model_version=model_version,
            lang_model=lang_model,
            device=self.device,
            predict=False,
        )
        if hasattr(self.model, "model"):
            self.model.model.eval()
        else:
            self.model.eval()

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode_prompts(prompts, onehot=False)
        return F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def encode_image_to_feature_map(self, img_tensor: torch.Tensor) -> torch.Tensor:
        feat_map = self.model.encode_image_to_feat_map(img_tensor.to(self.device))
        aligned = self.model.align_spatial_features_with_language(feat_map, onehot=False)
        return F.normalize(aligned, dim=1)


class Talk2DINOBackend(VisionLanguageBackend):
    def __init__(self, model_id="lorebianchi98/Talk2DINO-ViTL", device="cuda"):
        super().__init__(device=device)
        print(f"Loading Talk2DINO model {model_id}...")
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        self.transform = T.Compose(
            [
                T.Resize((448, 448)),
                T.ToTensor(),
            ]
        )

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        outputs = []
        for prompt in prompts:
            text_embed = self.model.encode_text(prompt)
            if not isinstance(text_embed, torch.Tensor):
                text_embed = torch.as_tensor(text_embed)
            if text_embed.dim() == 1:
                text_embed = text_embed.unsqueeze(0)
            outputs.append(text_embed.to(self.device))
        embeddings = torch.cat(outputs, dim=0)
        return F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def encode_image_to_feature_map(self, img_tensor: torch.Tensor) -> torch.Tensor:
        image_embed = self.model.encode_image(img_tensor.to(self.device))
        if not isinstance(image_embed, torch.Tensor):
            image_embed = torch.as_tensor(image_embed)
        image_embed = image_embed.to(self.device)

        if image_embed.dim() == 4:
            if image_embed.shape[1] <= 8 and image_embed.shape[-1] > image_embed.shape[1]:
                # (B, H, W, C)
                feature_map = image_embed.permute(0, 3, 1, 2)
            else:
                feature_map = image_embed
        elif image_embed.dim() == 3:
            batch_size, num_tokens, channels = image_embed.shape
            side = int(math.isqrt(num_tokens))
            if side * side != num_tokens:
                raise ValueError(f"Talk2DINO patch token count {num_tokens} is not a square grid.")
            feature_map = image_embed.reshape(batch_size, side, side, channels).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported Talk2DINO image embedding shape: {tuple(image_embed.shape)}")

        return F.normalize(feature_map, dim=1)


def create_backend(
    backend_name,
    device="cuda",
    model_version="c-radio_v4-h",
    lang_model="siglip2-g",
    model_id=None,
):
    backend_name = backend_name.lower()
    if backend_name == "radseg":
        return RADSegBackend(model_version=model_version, lang_model=lang_model, device=device)
    if backend_name == "talk2dino":
        resolved_model_id = model_id or "lorebianchi98/Talk2DINO-ViTL"
        return Talk2DINOBackend(model_id=resolved_model_id, device=device)
    raise ValueError(f"Unsupported backend '{backend_name}'")
