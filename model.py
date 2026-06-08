import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableQueries(nn.Module):
    """
    Co2S-style class-wise learnable queries.
    For this anomaly segmentation codebase, num_queries should match the
    number of semantic classes (normal / anomaly => 2).
    """

    def __init__(self, num_queries: int, embed_dim: int, freeze: bool = False):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        nn.init.normal_(self.query_embed.weight, mean=0.0, std=0.02)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


class DINOBackboneWrapper(nn.Module):
    """
    Wrapper for DINO-style ViT backbones.

    Preferred loading order:
      1) Official DINOv3 hub entrypoint with local weights path
      2) timm backbone fallback

    Supported aliases:
      - vit_base_patch16_224 -> dinov3_vitb16
      - vit_large_patch16_224 -> dinov3_vitl16
    """

    _OFFICIAL_NAME_ALIASES = {
        "vit_base_patch16_224": "dinov3_vitb16",
        "vit_large_patch16_224": "dinov3_vitl16",
        "vitb16": "dinov3_vitb16",
        "vitl16": "dinov3_vitl16",
    }

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        pretrained: bool = True,
        ckpt_path: str = "",
        source: str = "auto",
        repo_or_dir: str = "facebookresearch/dinov3",
        repo_source: str = "github",
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.ckpt_path = ckpt_path
        self.source = source
        self.repo_or_dir = repo_or_dir
        self.repo_source = repo_source
        self.backbone = self._build_backbone()
        self.embed_dim = self._infer_embed_dim()

    def _resolve_official_name(self) -> Optional[str]:
        if self.model_name.startswith("dinov3_"):
            return self.model_name
        return self._OFFICIAL_NAME_ALIASES.get(self.model_name)

    @staticmethod
    def _unwrap_state_dict(state):
        if not isinstance(state, dict):
            return state
        for key in ["state_dict", "model", "teacher", "student", "module"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
        if not isinstance(state, dict):
            return state

        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ("module.", "backbone.", "model.", "teacher.", "student."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        return cleaned

    def _load_local_checkpoint(self, model: nn.Module, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location="cpu")
        state = self._unwrap_state_dict(state)
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {type(state)}")

        model_state = model.state_dict()
        filtered_state = {
            k: v for k, v in state.items()
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
        }
        if len(filtered_state) == 0:
            raise RuntimeError(
                "No checkpoint keys matched the created model. "
                f"model_name={self.model_name}, ckpt_path={ckpt_path}"
            )

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(
            f"[DINOBackboneWrapper] Loaded local checkpoint from {ckpt_path}; "
            f"matched={len(filtered_state)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def _build_backbone(self):
        errors = []
        official_name = self._resolve_official_name()
        resolved_ckpt_path = self.ckpt_path if self.ckpt_path and os.path.isfile(self.ckpt_path) else self.ckpt_path

        if self.source in ("local", "official", "torchhub", "auto") and official_name is not None:
            try:
                hub_kwargs = {}
                if resolved_ckpt_path:
                    hub_kwargs["weights"] = resolved_ckpt_path
                model = torch.hub.load(
                    self.repo_or_dir,
                    official_name,
                    source=self.repo_source,
                    **hub_kwargs,
                )
                print(
                    f"[DINOBackboneWrapper] Built official DINOv3 backbone: "
                    f"entry={official_name}, repo={self.repo_or_dir}, repo_source={self.repo_source}"
                )
                return model
            except Exception as e:  # pragma: no cover - runtime dependent
                errors.append(f"official_dinov3={repr(e)}")

        if self.source in ("timm", "auto"):
            try:
                import timm  # type: ignore
                model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=0)
                if resolved_ckpt_path:
                    self._load_local_checkpoint(model, resolved_ckpt_path)
                print(f"[DINOBackboneWrapper] Built timm backbone: model_name={self.model_name}")
                return model
            except Exception as e:  # pragma: no cover - runtime dependent
                errors.append(f"timm={repr(e)}")

        raise RuntimeError(
            "Unable to build DINO backbone. "
            f"model_name={self.model_name}, source={self.source}, repo_or_dir={self.repo_or_dir}, "
            f"repo_source={self.repo_source}, resolved_ckpt_path={resolved_ckpt_path}, errors={' | '.join(errors)}"
        )

    def _infer_embed_dim(self) -> int:
        for attr in ["embed_dim", "num_features", "feature_dim"]:
            if hasattr(self.backbone, attr):
                value = getattr(self.backbone, attr)
                if isinstance(value, int):
                    return value
        return 768

    def _extract_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)

        if isinstance(feats, dict):
            if "x_norm_clstoken" in feats and "x_norm_patchtokens" in feats:
                return feats["x_norm_clstoken"], feats["x_norm_patchtokens"]
            if "cls_token" in feats and "patch_tokens" in feats:
                return feats["cls_token"], feats["patch_tokens"]
            if "x_prenorm" in feats and isinstance(feats["x_prenorm"], torch.Tensor):
                token_seq = feats["x_prenorm"]
            elif "x" in feats and isinstance(feats["x"], torch.Tensor) and feats["x"].dim() == 3:
                token_seq = feats["x"]
            else:
                token_seq = None
                for v in feats.values():
                    if isinstance(v, torch.Tensor) and v.dim() == 3:
                        token_seq = v
                        break
                if token_seq is None:
                    raise RuntimeError("Unsupported DINO forward_features dict output")
        else:
            token_seq = feats

        if token_seq.dim() == 2:
            return token_seq, token_seq.unsqueeze(1)

        if token_seq.dim() != 3:
            raise RuntimeError(f"Unsupported token sequence shape: {tuple(token_seq.shape)}")

        n = token_seq.shape[1]
        if int(math.sqrt(n - 1)) ** 2 == n - 1:
            image_token = token_seq[:, 0]
            patch_tokens = token_seq[:, 1:]
        elif int(math.sqrt(n)) ** 2 == n:
            image_token = token_seq.mean(dim=1)
            patch_tokens = token_seq
        else:
            image_token = token_seq.mean(dim=1)
            patch_tokens = token_seq[:, 1:] if n > 1 else token_seq
        return image_token, patch_tokens

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._extract_tokens(x)


class DINOv3DetailBranch(nn.Module):
    """
    Query-guided DINO detail branch.

    Borrowing the Co2S idea, this branch uses class-wise learnable queries for
    implicit guidance and keeps the entire branch in logit space.
    """

    def __init__(
        self,
        dino_model_name: str,
        dino_source: str,
        dino_pretrained: bool,
        dino_ckpt_path: str,
        text_embed_dim: int,
        dino_repo_or_dir: str = "facebookresearch/dinov3",
        dino_repo_source: str = "github",
        num_queries: int = 2,
        attn_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        del text_embed_dim  # kept only for CLI/checkpoint compatibility
        self.backbone = DINOBackboneWrapper(
            model_name=dino_model_name,
            pretrained=dino_pretrained,
            ckpt_path=dino_ckpt_path,
            source=dino_source,
            repo_or_dir=dino_repo_or_dir,
            repo_source=dino_repo_source,
        )
        self.embed_dim = self.backbone.embed_dim
        self.num_queries = num_queries
        self.learnable_queries = LearnableQueries(num_queries, self.embed_dim)
        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.query_cross_attn = nn.MultiheadAttention(self.embed_dim, attn_heads, dropout=dropout, batch_first=True)
        self.patch_cross_attn = nn.MultiheadAttention(self.embed_dim, attn_heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.patch_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        # Re-normalization buffers: CLIP -> DINO (ImageNet)
        self.register_buffer('_clip_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('_clip_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.register_buffer('_dino_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_dino_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _renormalize_clip_to_dino(self, image: torch.Tensor) -> torch.Tensor:
        """Convert CLIP-normalized image to DINO (ImageNet) normalization."""
        img = image * self._clip_std + self._clip_mean
        return (img - self._dino_mean) / self._dino_std

    def forward(self, image: torch.Tensor, image_size: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        image = self._renormalize_clip_to_dino(image)
        image_token, patch_tokens = self.backbone(image)
        batch_size, num_patches, _ = patch_tokens.shape
        side = int(math.sqrt(num_patches))
        if side * side != num_patches:
            raise RuntimeError(f"DINO patch token count {num_patches} is not a square grid.")

        patch_tokens = F.normalize(patch_tokens, dim=-1)
        image_token = F.normalize(image_token, dim=-1)

        queries = self.learnable_queries(batch_size)
        queries = self.query_proj(queries)
        query_context, _ = self.query_cross_attn(queries, patch_tokens, patch_tokens, need_weights=False)
        guided_queries = self.query_norm(queries + self.dropout(query_context))

        patch_context, _ = self.patch_cross_attn(patch_tokens, guided_queries, guided_queries, need_weights=False)
        guided_patches = self.patch_norm(patch_tokens + self.dropout(patch_context))

        guided_queries = F.normalize(guided_queries, dim=-1)
        guided_patches = F.normalize(guided_patches, dim=-1)
        image_token = F.normalize(image_token, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)

        pixel_logits_tokens = scale * torch.einsum("bnc,bkc->bnk", guided_patches, guided_queries)
        pixel_logits = pixel_logits_tokens.reshape(batch_size, side, side, self.num_queries).permute(0, 3, 1, 2)
        pixel_logits = F.interpolate(pixel_logits, size=(image_size, image_size), mode="bilinear", align_corners=False)

        image_logits = scale * torch.einsum("bc,bkc->bk", image_token, guided_queries)

        aux = {
            "guided_queries": guided_queries.detach(),
            "guided_patches": guided_patches.detach(),
        }
        return pixel_logits, image_logits, aux




class NaiveFusionHead(nn.Module):
    

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, clip_logits: torch.Tensor, dino_logits: torch.Tensor) -> torch.Tensor:
        clip_prob = torch.softmax(clip_logits, dim=-1)
        dino_prob = torch.softmax(dino_logits, dim=-1)
        fused_prob = 0.5 * clip_prob + 0.5 * dino_prob
        return torch.log(fused_prob.clamp(min=1e-8))


class SpatialPriorFiLMModulator(nn.Module):
    """
    Spatially-Adaptive FiLM modulation (SPADE-like): use Stage 1 dense spatial prior (guided_patches)
    to produce per-channel and per-spatial-location scale/shift for CLIP patch features.

    Initialized near-identity so early training preserves CLIP patch quality.
    """

    def __init__(self, prior_dim: int, patch_dim: int, prior_dropout: float = 0.3):
        super().__init__()
        self.prior_dropout = nn.Dropout(prior_dropout)
        self.prior_norm = nn.LayerNorm(prior_dim)
        
        # We process spatially with 3x3 Conv
        self.gamma_net = nn.Sequential(
            nn.Conv2d(prior_dim, patch_dim, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.beta_net = nn.Sequential(
            nn.Conv2d(prior_dim, patch_dim, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        
        # Near-identity init: gamma ~ 1 (sigmoid(3) ~ 0.95), beta ~ 0
        nn.init.zeros_(self.gamma_net[0].weight)
        nn.init.constant_(self.gamma_net[0].bias, 3.0)
        nn.init.zeros_(self.beta_net[0].weight)
        nn.init.zeros_(self.beta_net[0].bias)

    def forward(self, patch_features: torch.Tensor, guided_patches: torch.Tensor) -> torch.Tensor:
        """
        patch_features:  (B, N_clip, D_patch) -- CLIP patch tokens (may lack grad).
        guided_patches:  (B, N_dino, D_prior) -- spatially distributed patch tokens from frozen Stage 1.
        Returns:         (B, N_clip, D_patch) -- modulated patch tokens.
        """
        import math
        B, N_clip, D_patch = patch_features.shape
        _, N_dino, D_prior = guided_patches.shape

        # Determine spatial dimensions
        H_clip = int(math.sqrt(N_clip))
        W_clip = H_clip
        if H_clip * W_clip != N_clip:
            raise RuntimeError(f"CLIP patch token count {N_clip} is not a square grid.")

        H_dino = int(math.sqrt(N_dino))
        W_dino = H_dino
        if H_dino * W_dino != N_dino:
            raise RuntimeError(f"DINO patch token count {N_dino} is not a square grid.")

        # Reshape to spatial format
        prior = guided_patches.detach().transpose(1, 2).contiguous().view(B, D_prior, H_dino, W_dino) # (B, D_prior, H_dino, W_dino)
        
        # Spatial interpolation to match CLIP resolution
        if H_dino != H_clip or W_dino != W_clip:
            prior = F.interpolate(prior, size=(H_clip, W_clip), mode='bilinear', align_corners=False)
        
        # Apply dropout and normalization. LayerNorm requires channels last for standard usage over channels.
        prior = prior.permute(0, 2, 3, 1).contiguous()   # (B, H_clip, W_clip, D_prior)
        prior = self.prior_dropout(prior)
        prior = self.prior_norm(prior)
        prior = prior.permute(0, 3, 1, 2).contiguous()   # (B, D_prior, H_clip, W_clip)
        
        gamma = self.gamma_net(prior)  # (B, D_patch, H_clip, W_clip)
        beta  = self.beta_net(prior)   # (B, D_patch, H_clip, W_clip)
        
        # Reshape back to sequence format for element-wise modulation
        gamma = gamma.flatten(2).transpose(1, 2)  # (B, N_clip, D_patch)
        beta  = beta.flatten(2).transpose(1, 2)   # (B, N_clip, D_patch)
        
        return gamma * patch_features + beta
