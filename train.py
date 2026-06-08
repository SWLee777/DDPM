import argparse
import os
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import AnomalyCLIP_lib
from model import (
    DINOv3DetailBranch,
    NaiveFusionHead,
    SpatialPriorFiLMModulator,
)
from dataset import Dataset
from logger import get_logger
from loss import BinaryDiceLoss, FocalLoss
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_module(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def compute_segmentation_loss(loss_focal, loss_dice, prediction_logits, gt, loss_scale=1.0):
    prediction_logits = prediction_logits.float()
    gt = gt.float()
    prob = torch.softmax(prediction_logits, dim=1).clamp(1e-6, 1.0 - 1e-6)
    loss = loss_focal(prob, gt)
    loss = loss + loss_dice(prob[:, 1, :, :], gt)
    loss = loss + loss_dice(prob[:, 0, :, :], 1 - gt)
    return loss_scale * loss


def prepare_gt_mask(gt: torch.Tensor, device: str) -> torch.Tensor:
    gt = gt.to(device)
    if gt.dim() == 4 and gt.shape[1] == 1:
        gt = gt[:, 0, :, :]
    elif gt.dim() == 2:
        gt = gt.unsqueeze(0)
    gt = gt.float()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0
    return gt


def encode_text_features(model, prompt_learner, device):
    del device
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def _resolve_selected_layers(num_layers: int, feature_map_layer) -> Iterable[int]:
    if feature_map_layer is None:
        return range(num_layers)
    if isinstance(feature_map_layer, int):
        return [feature_map_layer]
    if len(feature_map_layer) == 0:
        return range(num_layers)
    return sorted({int(idx) for idx in feature_map_layer if 0 <= int(idx) < num_layers})


def semantic_forward(model, text_features, image, features_list, feature_map_layer, image_size, layer=20):
    image_features, patch_features = model.encode_image(image, features_list, DPAM_layer=dpam_layer)
    image_features = F.normalize(image_features, dim=-1)
    image_logits = (image_features.unsqueeze(1) @ text_features.permute(0, 2, 1))[:, 0, ...] / 0.07

    semantic_logits_per_layer = []
    selected_layers = _resolve_selected_layers(len(patch_features), feature_map_layer)
    for idx in selected_layers:
        patch_feature = F.normalize(patch_features[idx], dim=-1)
        similarity_logits = (patch_feature @ text_features[0].t()) / 0.07
        semantic_logits = AnomalyCLIP_lib.get_similarity_map(
            similarity_logits[:, 1:, :],
            image_size,
        ).permute(0, 3, 1, 2)
        semantic_logits_per_layer.append(semantic_logits)

    if not semantic_logits_per_layer:
        raise RuntimeError("No semantic feature maps were selected for supervision / inference.")

    semantic_map_logits = torch.stack(semantic_logits_per_layer, dim=0).mean(dim=0)
    return semantic_map_logits, image_logits, semantic_logits_per_layer


def semantic_forward_with_prior(model, text_features, image, features_list,
                                feature_map_layer, image_size,
                                prior_modulator, guided_patches, dpam_layer=20):
    """CLIP patch pipeline with Spatial FiLM prior injection from Stage 1."""
    image_features, patch_features = model.encode_image(image, features_list, DPAM_layer=dpam_layer)
    image_features = F.normalize(image_features, dim=-1)
    image_logits = (image_features.unsqueeze(1) @ text_features.permute(0, 2, 1))[:, 0, ...] / 0.07

    pixel_maps = []
    selected_layers = _resolve_selected_layers(len(patch_features), feature_map_layer)
    for idx in selected_layers:
        patch_feat = patch_features[idx][:, 1:, :]  # (B, N, D), remove CLS token

        # Spatial FiLM modulation with Stage 1 prior
        patch_feat = prior_modulator(patch_feat, guided_patches)
        patch_feat = F.normalize(patch_feat, dim=-1)

        sim = (patch_feat @ text_features[0].t()) / 0.07  # (B, N, 2)
        pixel_map = AnomalyCLIP_lib.get_similarity_map(sim, image_size).permute(0, 3, 1, 2)
        pixel_maps.append(pixel_map)

    if not pixel_maps:
        raise RuntimeError("No feature maps selected for pixel prediction.")

    fused_pixel_map = torch.stack(pixel_maps, dim=0).mean(dim=0)
    return fused_pixel_map, image_logits, pixel_maps


def train_stage1_image_only(args):
    """
    Stage 1: Train CLIP + DINO for image-level classification.
    Fusion: NaiveFusionHead (probability averaging, no learnable params).
    """
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_cfg = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }
    model_clip, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=clip_cfg)
    model_clip.eval()
    freeze_module(model_clip)
    model_clip.visual.DAPM_replace(DPAM_layer=20)

    stage1_prompt_learner = AnomalyCLIP_PromptLearner(model_clip.to("cpu"), clip_cfg).to(device)
    model_clip.to(device)

    detail_model = DINOv3DetailBranch(
        dino_model_name=args.dino_model_name,
        dino_source=args.dino_source,
        dino_pretrained=args.dino_pretrained,
        dino_ckpt_path=args.dino_ckpt_path,
        text_embed_dim=model_clip.text_projection.shape[1],
        dino_repo_or_dir=args.dino_repo_or_dir,
        dino_repo_source=args.dino_repo_source,
        num_queries=args.detail_num_queries,
        attn_heads=args.detail_attn_heads,
        dropout=args.detail_dropout,
    ).to(device)

    fusion_head = NaiveFusionHead().to(device)

    if not args.train_dino_backbone:
        freeze_module(detail_model.backbone)

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # NaiveFusionHead has no learnable parameters, excluded from optimizer
    optimizer = torch.optim.Adam(
        list(stage1_prompt_learner.parameters())
        + [p for p in detail_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    for epoch in tqdm(range(args.epoch)):
        clip_losses, dino_losses, fused_losses, total_losses = [], [], [], []

        for items in tqdm(train_loader):
            image = items["img"].to(device)
            label = items["anomaly"].to(device)

            text_features = encode_text_features(model_clip, stage1_prompt_learner, device)
            image_features, _ = model_clip.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = F.normalize(image_features, dim=-1)
            clip_image_logits = (image_features.unsqueeze(1) @ text_features.permute(0, 2, 1))[:, 0, ...] / 0.07

            _, dino_image_logits, _ = detail_model(image=image, image_size=args.image_size)

            fused_image_logits = fusion_head(clip_image_logits, dino_image_logits)

            loss_clip = F.cross_entropy(clip_image_logits, label.long())
            loss_dino = F.cross_entropy(dino_image_logits, label.long())
            loss_fused = F.cross_entropy(fused_image_logits, label.long())

            total_loss = (
                args.stage1_clip_weight * loss_clip
                + args.stage1_dino_weight * loss_dino
                +  loss_fused
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            clip_losses.append(loss_clip.item())
            dino_losses.append(loss_dino.item())
            fused_losses.append(loss_fused.item())
            total_losses.append(total_loss.item())

        logger.info(
            "Stage1 epoch [%d/%d], clip_img:%.4f, dino_img:%.4f, fused_img:%.4f, total:%.4f",
            epoch + 1, args.epoch,
            np.mean(clip_losses), np.mean(dino_losses),
            np.mean(fused_losses), np.mean(total_losses),
        )

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, f"epoch_{epoch + 1}.pth")
            torch.save({
                "stage1_prompt_learner": stage1_prompt_learner.state_dict(),
                "detail_model": detail_model.state_dict(),
                "training_stage": "stage1",
                "args": vars(args),
            }, ckp_path)


def train_stage2_pixel_with_prior(args):
    """
    Stage 2: Train pixel-level predictor with Stage 1 DINO spatial prior.
    Trainable: stage2_prompt_learner + SpatialPriorFiLMModulator.
    """
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_cfg = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }
    model_clip, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=clip_cfg)
    model_clip.eval()
    freeze_module(model_clip)
    model_clip.visual.DAPM_replace(DPAM_layer=20)

    # Load Stage 1 detail model (frozen, for guided_patches)
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location=device)
    detail_model = DINOv3DetailBranch(
        dino_model_name=args.dino_model_name,
        dino_source=args.dino_source,
        dino_pretrained=args.dino_pretrained,
        dino_ckpt_path=args.dino_ckpt_path,
        text_embed_dim=model_clip.text_projection.shape[1],
        dino_repo_or_dir=args.dino_repo_or_dir,
        dino_repo_source=args.dino_repo_source,
        num_queries=args.detail_num_queries,
        attn_heads=args.detail_attn_heads,
        dropout=args.detail_dropout,
    ).to(device)
    detail_model.load_state_dict(stage1_ckpt["detail_model"])
    detail_model.eval()
    freeze_module(detail_model)

    # Stage 2: NEW independent prompt learner (trainable)
    stage2_prompt_learner = AnomalyCLIP_PromptLearner(model_clip.to("cpu"), clip_cfg).to(device)
    model_clip.to(device)

    # Stage 2: Spatial FiLM prior modulator (trainable)
    prior_modulator = SpatialPriorFiLMModulator(
        prior_dim=detail_model.embed_dim,
        patch_dim=model_clip.text_projection.shape[1],
        prior_dropout=args.prior_dropout,
    ).to(device)

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        list(stage2_prompt_learner.parameters())
        + list(prior_modulator.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    lam = 4.0

    for epoch in tqdm(range(args.epoch)):
        seg_losses, img_losses, total_losses = [], [], []

        for items in tqdm(train_loader):
            image = items["img"].to(device)
            label = items["anomaly"].to(device)
            gt = prepare_gt_mask(items["img_mask"], device)

            # Stage 1 frozen forward -> guided_patches
            with torch.no_grad():
                _, _, aux = detail_model(image=image, image_size=args.image_size)
                guided_patches = aux["guided_patches"]

            # Stage 2: CLIP patches + Spatial FiLM modulation
            text_features = encode_text_features(model_clip, stage2_prompt_learner, device)
            pixel_map, image_logits, pixel_maps_per_layer = semantic_forward_with_prior(
                model_clip, text_features, image,
                args.features_list, args.feature_map_layer, args.image_size,
                prior_modulator, guided_patches,
            )

            # Per-layer + fused pixel loss
            seg_loss = 0.0
            for pm in pixel_maps_per_layer:
                seg_loss = seg_loss + compute_segmentation_loss(loss_focal, loss_dice, pm, gt)
            seg_loss = lam * seg_loss / len(pixel_maps_per_layer)

            # Auxiliary image CE loss (anchors prompt learner semantics)
            image_loss = args.stage2_aux_img_weight * F.cross_entropy(image_logits, label.long())

            total_loss = seg_loss + image_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            seg_losses.append(seg_loss.item())
            img_losses.append(image_loss.item())
            total_losses.append(total_loss.item())

        logger.info(
            "Stage2 epoch [%d/%d], seg:%.4f, aux_img:%.4f, total:%.4f",
            epoch + 1, args.epoch,
            np.mean(seg_losses), np.mean(img_losses), np.mean(total_losses),
        )

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, f"epoch_{epoch + 1}.pth")
            torch.save({
                "stage2_prompt_learner": stage2_prompt_learner.state_dict(),
                "prior_modulator": prior_modulator.state_dict(),
                "training_stage": "stage2",
                "stage1_ckpt_path": args.stage1_ckpt,
                "args": vars(args),
            }, ckp_path)


def train(args):
    os.makedirs(args.save_path, exist_ok=True)
    if args.training_stage == "stage1":
        train_stage1_image_only(args)
    elif args.training_stage == "stage2":
        train_stage2_pixel_with_prior(args)
    else:
        raise ValueError(f"Unknown training_stage: {args.training_stage}. Use 'stage1' or 'stage2'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="", help="path to train dataset")
    parser.add_argument("--save_path", type=str, default="./checkpoint", help="path to save results")
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111)

    # DINO branch
    parser.add_argument("--dino_model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--dino_source", type=str, default="local", choices=["local", "official", "timm", "torchhub", "auto"])
    parser.add_argument("--dino_pretrained", action="store_true")
    parser.add_argument("--dino_ckpt_path", type=str, default="",help="path to dino checkpoint")
    parser.add_argument("--dino_repo_or_dir", type=str, default="",help="path to dino repo or directory")
    parser.add_argument("--dino_repo_source", type=str, default="local", choices=["github", "local"])
    parser.add_argument("--train_dino_backbone", action="store_true")
    parser.add_argument("--detail_num_queries", type=int, default=2)
    parser.add_argument("--detail_attn_heads", type=int, default=8)
    parser.add_argument("--detail_dropout", type=float, default=0.1)

    # Two-stage
    parser.add_argument("--training_stage", type=str, required=True, choices=["stage1", "stage2"])
    parser.add_argument("--stage1_ckpt", type=str, default="", help="Path to Stage 1 checkpoint (required for stage2)")
    parser.add_argument("--prior_dropout", type=float, default=0.3, help="Dropout for FiLM prior modulator (Stage 2)")
    parser.add_argument("--stage1_clip_weight", type=float, default=1.0)
    parser.add_argument("--stage1_dino_weight", type=float, default=1.0)
    parser.add_argument("--stage2_aux_img_weight", type=float, default=0.1)

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
