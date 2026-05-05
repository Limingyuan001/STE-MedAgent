from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from timm.models.layers import DropPath, Mlp, trunc_normal_
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor


BIOVIL_T_MODEL_NAME = "biovil-t-image"
BIOVIL_T_HF_REPO = "microsoft/BiomedVLP-BioViL-T"
BIOVIL_T_WEIGHTS_NAME = "biovil_t_image_model_proj_size_128.pt"
BIOVIL_T_RESIZE = 512
BIOVIL_T_CENTER_CROP = 448
BIOVIL_T_JOINT_FEATURE_SIZE = 128


class ExpandChannels:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {tuple(data.shape)}")
        return torch.repeat_interleave(data, 3, dim=0)


class ResNetHIML(ResNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4


def resnet50(pretrained: bool = False, **kwargs: object) -> ResNetHIML:
    del pretrained
    return ResNetHIML(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)


def get_module_device(module: torch.nn.Module) -> torch.device:
    device = next(module.parameters()).device
    assert isinstance(device, torch.device)
    return device


@dataclass
class ImageModelOutput:
    img_embedding: torch.Tensor
    patch_embeddings: torch.Tensor
    projected_global_embedding: torch.Tensor
    projected_patch_embeddings: torch.Tensor


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        use_1x1_convs: bool = False,
    ) -> None:
        super().__init__()

        if use_1x1_convs:
            linear_proj_1_args = {
                "in_channels": input_dim,
                "out_channels": hidden_dim,
                "kernel_size": 1,
                "bias": False,
            }
            linear_proj_2_args = {
                "in_channels": hidden_dim,
                "out_channels": output_dim,
                "kernel_size": 1,
                "bias": True,
            }
            normalisation_layer = nn.BatchNorm2d
            projection_layer = nn.Conv2d
        else:
            linear_proj_1_args = {
                "in_features": input_dim,
                "out_features": hidden_dim,
                "bias": False,
            }
            linear_proj_2_args = {
                "in_features": hidden_dim,
                "out_features": output_dim,
                "bias": True,
            }
            normalisation_layer = nn.BatchNorm1d
            projection_layer = nn.Linear

        if hidden_dim is not None:
            self.model = nn.Sequential(
                projection_layer(**linear_proj_1_args),
                normalisation_layer(hidden_dim),
                nn.ReLU(inplace=True),
                projection_layer(**linear_proj_2_args),
            )
        else:
            self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class MultiHeadAttentionOutput:
    mha_output: torch.Tensor
    attention: Optional[torch.Tensor] = None


class SinePositionEmbedding:
    def __init__(
        self,
        embedding_dim: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * torch.pi if scale is None else scale

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            raise ValueError("No pixel mask provided")
        _, height, width = mask.shape
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos.view(1, height * width, -1)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Embedding dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.return_attention = False

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward_as_mhsa(self, input: torch.Tensor) -> MultiHeadAttentionOutput:
        batch_size, num_tokens, channels = input.shape
        query = self.proj_q(input).reshape(batch_size, num_tokens, self.num_heads, channels // self.num_heads)
        key = self.proj_k(input).reshape(batch_size, num_tokens, self.num_heads, channels // self.num_heads)
        value = self.proj_v(input).reshape(batch_size, num_tokens, self.num_heads, channels // self.num_heads)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        output = (attn @ value).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        output = self.proj(output)
        output = self.proj_drop(output)
        return MultiHeadAttentionOutput(
            mha_output=output,
            attention=attn if self.return_attention else None,
        )


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 1.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttentionLayer(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    @staticmethod
    def with_pos_and_type_embed(tensor: torch.Tensor, emb: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if emb is None else tensor + emb

    def forward(self, x: torch.Tensor, pos_and_type_embed: Optional[torch.Tensor]) -> torch.Tensor:
        x_with_emb = self.with_pos_and_type_embed(self.norm1(x), emb=pos_and_type_embed)
        x = x + self.drop_path(self.attn.forward_as_mhsa(x_with_emb).mha_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformerPooler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        grid_shape: Tuple[int, int],
        num_heads: int = 8,
        num_blocks: int = 3,
        norm_layer: type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        block_kwargs = dict(
            dim=input_dim,
            num_heads=num_heads,
            mlp_ratio=1.0,
            drop=0.10,
            attn_drop=0.10,
            drop_path=0.25,
            act_layer=nn.GELU,
            norm_layer=norm_layer,
        )
        self.blocks = nn.ModuleList([Block(**block_kwargs) for _ in range(num_blocks)])
        self.norm_post = norm_layer(input_dim)
        self.grid_shape = grid_shape
        self.num_patches = grid_shape[0] * grid_shape[1]
        self.type_embed = nn.Parameter(torch.zeros(2, 1, input_dim))
        trunc_normal_(self.type_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=0.10)
        pos_embed_class = SinePositionEmbedding(embedding_dim=input_dim // 2, normalize=True)
        pos_embed = pos_embed_class(mask=torch.ones([1, grid_shape[0], grid_shape[1]]))
        self.register_buffer("pos_embed", pos_embed, persistent=False)
        self.apply(self._init_weights)

    def forward(self, current_image: torch.Tensor, previous_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, channels, height, width = current_image.shape
        if (height, width) != self.grid_shape:
            raise ValueError(f"Expected grid {self.grid_shape}, got {(height, width)}")
        current_image = current_image.view(batch_size, channels, height * width).transpose(1, 2)
        previous_tokens = None
        if previous_image is not None:
            if previous_image.shape != current_image.transpose(1, 2).view(batch_size, channels, height, width).shape:
                raise ValueError("current_image and previous_image shapes do not match")
            previous_tokens = previous_image.view(batch_size, channels, height * width).transpose(1, 2)
        pos_embed = self.pos_embed.repeat(batch_size, 1, 1)
        token_features = self.forward_after_reshape(current_image, pos_embed, previous_tokens)
        current_token_features = token_features[:, : self.num_patches]
        return current_token_features.transpose(1, 2).view(batch_size, channels, height, width)

    def forward_after_reshape(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        x_previous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        type_embed = self.type_embed[0].expand(batch_size, seq_len, -1)
        if x_previous is not None:
            x = torch.cat((x, x_previous), dim=1)
            pos_embed = torch.cat((pos_embed, pos_embed), dim=1)
            prev_type_embed = self.type_embed[1].expand(batch_size, seq_len, -1)
            type_embed = torch.cat((type_embed, prev_type_embed), dim=1)

        pos_and_type_embed = pos_embed + type_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x=x, pos_and_type_embed=pos_and_type_embed)
        return self.norm_post(x)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = resnet50(pretrained=False)

    def forward(self, current_image: torch.Tensor, return_patch_embeddings: bool = False):
        patch_emb = self.encoder(current_image)
        avg_pooled_emb = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(patch_emb, (1, 1)), 1)
        if return_patch_embeddings:
            return patch_emb, avg_pooled_emb
        return avg_pooled_emb


class MultiImageEncoder(ImageEncoder):
    def __init__(self) -> None:
        super().__init__()
        output_dim = 256
        grid_shape = (14, 14)
        backbone_output_feature_dim = get_encoder_output_dim(self.encoder, device=get_module_device(self.encoder))
        self.backbone_to_vit = nn.Conv2d(
            in_channels=backbone_output_feature_dim,
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.vit_pooler = VisionTransformerPooler(input_dim=output_dim, grid_shape=grid_shape)
        self.missing_previous_emb = nn.Parameter(torch.zeros(1, output_dim, 1, 1))
        trunc_normal_(self.missing_previous_emb, std=0.02)

    def forward(
        self,
        current_image: torch.Tensor,
        previous_image: Optional[torch.Tensor] = None,
        return_patch_embeddings: bool = False,
    ):
        batch_size = current_image.shape[0]
        if previous_image is not None:
            if current_image.shape != previous_image.shape:
                raise ValueError("current_image and previous_image shapes do not match")
            x = torch.cat([current_image, previous_image], dim=0)
            x = super().forward(x, return_patch_embeddings=True)[0]
            x = self.backbone_to_vit(x)
            patch_x, patch_x_previous = x[:batch_size], x[batch_size:]
            diff_x = self.vit_pooler(current_image=patch_x, previous_image=patch_x_previous)
        else:
            x = super().forward(current_image, return_patch_embeddings=True)[0]
            patch_x = self.backbone_to_vit(x)
            batch_size, _, width, height = patch_x.shape
            diff_x = self.missing_previous_emb.repeat(batch_size, 1, width, height)

        patch_fused = torch.cat([patch_x, diff_x], dim=1)
        avg_pooled_emb = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(patch_fused, (1, 1)), 1)
        if return_patch_embeddings:
            return patch_fused, avg_pooled_emb
        return avg_pooled_emb


@torch.no_grad()
def get_encoder_output_dim(module: torch.nn.Module, device: torch.device) -> int:
    x = torch.rand((1, 3, BIOVIL_T_CENTER_CROP, BIOVIL_T_CENTER_CROP), device=device)
    training_mode = module.training
    module.eval()
    representations = module(x)
    module.train(mode=training_mode)
    return int(representations.shape[1])


class ImageModel(nn.Module):
    def __init__(self, pretrained_model_path: Optional[Union[str, Path]] = None) -> None:
        super().__init__()
        self.encoder = MultiImageEncoder()
        self.feature_size = get_encoder_output_dim(self.encoder, device=get_module_device(self.encoder))
        self.projector = MLP(
            input_dim=self.feature_size,
            output_dim=BIOVIL_T_JOINT_FEATURE_SIZE,
            hidden_dim=BIOVIL_T_JOINT_FEATURE_SIZE,
            use_1x1_convs=True,
        )
        if pretrained_model_path is not None:
            state_dict = torch.load(pretrained_model_path, map_location="cpu")
            self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> ImageModelOutput:
        patch_x, pooled_x = self.encoder(x, return_patch_embeddings=True)
        projected_patch_embeddings = self.projector(patch_x)
        projected_global_embedding = torch.mean(projected_patch_embeddings, dim=(2, 3))
        return ImageModelOutput(
            img_embedding=pooled_x,
            patch_embeddings=patch_x,
            projected_patch_embeddings=projected_patch_embeddings,
            projected_global_embedding=projected_global_embedding,
        )


class BioViLTImageInferenceEngine:
    def __init__(self, cache_dir: str, device: str) -> None:
        weights_path = hf_hub_download(
            repo_id=BIOVIL_T_HF_REPO,
            filename=BIOVIL_T_WEIGHTS_NAME,
            cache_dir=cache_dir,
        )
        self.model = ImageModel(pretrained_model_path=weights_path).to(device)
        self.model.eval()
        self.transform = Compose(
            [
                Resize(BIOVIL_T_RESIZE),
                CenterCrop(BIOVIL_T_CENTER_CROP),
                ToTensor(),
                ExpandChannels(),
            ]
        )

    def load_and_transform_input_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        image = Image.open(image_path).convert("L")
        device = get_module_device(self.model)
        return self.transform(image).unsqueeze(0).to(device)

    @torch.no_grad()
    def get_projected_global_embedding(self, image_path: Union[str, Path]) -> torch.Tensor:
        input_image = self.load_and_transform_input_image(image_path)
        projected_img_emb = self.model.forward(input_image).projected_global_embedding
        projected_img_emb = F.normalize(projected_img_emb, dim=-1)
        if projected_img_emb.shape[0] != 1 or projected_img_emb.ndim != 2:
            raise ValueError(f"Unexpected BioViL-T embedding shape: {tuple(projected_img_emb.shape)}")
        return projected_img_emb[0]
