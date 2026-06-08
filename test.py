import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from tqdm import tqdm

import AnomalyCLIP_lib
from model import DINOv3DetailBranch, NaiveFusionHead, SpatialPriorFiLMModulator
from dataset import Dataset
from logger import get_logger
from metrics import image_level_metrics, pixel_level_metrics
from prompt_ensemble import AnomalyCLIP_PromptLearner
from train import encode_text_features, freeze_module, semantic_forward_with_prior
from utils import get_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(results, obj_list):
    rows = []
    img_auroc_ls, img_ap_ls, pix_auroc_ls, pix_aupro_ls = [], [], [], []

    for obj in obj_list:
        results[obj]["imgs_masks"] = torch.cat(results[obj]["imgs_masks"])
        results[obj]["anomaly_maps"] = torch.cat(results[obj]["anomaly_maps"]).detach().cpu().numpy()

        img_auroc = image_level_metrics(results, obj, "image-auroc")
        img_ap = image_level_metrics(results, obj, "image-ap")
        pix_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
        pix_aupro = pixel_level_metrics(results, obj, "pixel-aupro")

        img_auroc_ls.append(img_auroc)
        img_ap_ls.append(img_ap)
        pix_auroc_ls.append(pix_auroc)
        pix_aupro_ls.append(pix_aupro)
        rows.append([obj, f"{img_auroc*100:.1f}", f"{img_ap*100:.1f}", f"{pix_auroc*100:.1f}", f"{pix_aupro*100:.1f}"])

    rows.append([
        "MEAN",
        f"{np.mean(img_auroc_ls)*100:.1f}",
        f"{np.mean(img_ap_ls)*100:.1f}",
        f"{np.mean(pix_auroc_ls)*100:.1f}",
        f"{np.mean(pix_aupro_ls)*100:.1f}",
    ])
    table = tabulate(rows, headers=["object", "Img-AUROC", "Img-AP", "Pix-AUROC", "Pix-AUPRO"], tablefmt="pipe")
    means = {
        "img_auroc": np.mean(img_auroc_ls),
        "img_ap": np.mean(img_ap_ls),
        "pix_auroc": np.mean(pix_auroc_ls),
        "pix_aupro": np.mean(pix_aupro_ls),
    }
    return table, means


def test_two_stage(args):
    """
    Two-stage inference (F-experiment):
      Stage 1 -> image-level score via CLIP + DINO NaiveFusion
      Stage 2 -> pixel-level anomaly map via SpatialPriorFiLMModulator
    """
    logger = get_logger(args.save_path)
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

    # Load checkpoints
    stage1_ckpt = torch.load(args.stage1_ckpt, map_location=device)
    stage2_ckpt = torch.load(args.checkpoint_path, map_location=device)

    # Create both prompt learners (need model on CPU for init)
    model_clip.to("cpu")
    stage1_prompt_learner = AnomalyCLIP_PromptLearner(model_clip, clip_cfg)
    stage1_prompt_learner.load_state_dict(stage1_ckpt["stage1_prompt_learner"])
    stage2_prompt_learner = AnomalyCLIP_PromptLearner(model_clip, clip_cfg)
    stage2_prompt_learner.load_state_dict(stage2_ckpt["stage2_prompt_learner"])
    model_clip.to(device)
    stage1_prompt_learner.to(device).eval()
    stage2_prompt_learner.to(device).eval()

    # Stage 1 detail model (frozen)
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

    # NaiveFusionHead (no learnable params)
    fusion_head = NaiveFusionHead().to(device)
    fusion_head.eval()

    # Stage 2 FiLM modulator
    prior_modulator = SpatialPriorFiLMModulator(
        prior_dim=detail_model.embed_dim,
        patch_dim=model_clip.text_projection.shape[1],
        prior_dropout=0.0,
    ).to(device)
    prior_modulator.load_state_dict(stage2_ckpt["prior_modulator"])
    prior_modulator.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list
    results = {obj: {"gt_sp": [], "pr_sp": [], "imgs_masks": [], "anomaly_maps": []} for obj in obj_list}

    # Pre-compute text features
    stage1_text = encode_text_features(model_clip, stage1_prompt_learner, device)
    stage2_text = encode_text_features(model_clip, stage2_prompt_learner, device)

    for items in tqdm(test_loader):
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"].clone()
        gt_mask[gt_mask > 0.5] = 1
        gt_mask[gt_mask <= 0.5] = 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(items["anomaly"].detach().cpu())

        with torch.no_grad():
            # Stage 1: image score via CLIP + DINO NaiveFusion
            image_features, _ = model_clip.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = F.normalize(image_features, dim=-1)
            clip_image_logits = (image_features.unsqueeze(1) @ stage1_text.permute(0, 2, 1))[:, 0, ...] / 0.07

            _, dino_image_logits, aux = detail_model(image=image, image_size=args.image_size)
            guided_patches = aux["guided_patches"]

            fused_image_logits = fusion_head(clip_image_logits, dino_image_logits)
            image_score = torch.softmax(fused_image_logits, dim=-1)[:, 1]

            # Stage 2: pixel map with Spatial FiLM prior
            pixel_map, _, _ = semantic_forward_with_prior(
                model_clip, stage2_text, image,
                args.features_list, args.feature_map_layer, args.image_size,
                prior_modulator, guided_patches,
            )
            anomaly_map = torch.softmax(pixel_map, dim=1)[:, 1, :, :]

            results[cls_name]["pr_sp"].extend(image_score.detach().cpu())
            anomaly_map = torch.stack(
                [torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()],
                dim=0,
            )
            results[cls_name]["anomaly_maps"].append(anomaly_map)

    table, means = evaluate(results, obj_list)
    logger.info("\n%s", table)
    logger.info(
        "SUMMARY Img-AUROC=%.2f Img-AP=%.2f Pix-AUROC=%.2f Pix-AUPRO=%.2f",
        means["img_auroc"] * 100, means["img_ap"] * 100,
        means["pix_auroc"] * 100, means["pix_aupro"] * 100,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--data_path", type=str, default="", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default="./results/", help="path to save results")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to Stage 2 checkpoint")
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--sigma", type=int, default=4)

    # DINO branch
    parser.add_argument("--dino_model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--dino_source", type=str, default="local", choices=["local", "official", "timm", "torchhub", "auto"])
    parser.add_argument("--dino_pretrained", action="store_true")
    parser.add_argument("--dino_ckpt_path", type=str, default="",help="path to dino checkpoint")
    parser.add_argument("--dino_repo_or_dir", type=str, default="",help="path to dino repo or directory")
    parser.add_argument("--dino_repo_source", type=str, default="github", choices=["github", "local"])
    parser.add_argument("--detail_num_queries", type=int, default=2)
    parser.add_argument("--detail_attn_heads", type=int, default=8)
    parser.add_argument("--detail_dropout", type=float, default=0.1)

    # Stage 1 checkpoint
    parser.add_argument("--stage1_ckpt", type=str, required=True, help="Path to Stage 1 checkpoint")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args.seed)
    test_two_stage(args)
