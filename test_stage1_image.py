import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm

import AnomalyCLIP_lib
from model import DINOv3DetailBranch, NaiveFusionHead
from dataset import Dataset
from logger import get_logger
from metrics import image_level_metrics
from prompt_ensemble import AnomalyCLIP_PromptLearner
from train import freeze_module
from utils import get_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_text_features(model, prompt_learner, device):
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    return text_features / text_features.norm(dim=-1, keepdim=True)


def evaluate_image_only(results, obj_list):
    table_ls = []
    image_auroc_list, image_ap_list = [], []

    for obj in obj_list:
        table = [obj]
        # match structure from evaluate() in test.py
        results[obj]["imgs_masks"] = torch.cat(results[obj]["imgs_masks"])
        results[obj]["anomaly_maps"] = torch.cat(results[obj]["anomaly_maps"]).detach().cpu().numpy()
        
        image_auroc = image_level_metrics(results, obj, "image-auroc")
        image_ap = image_level_metrics(results, obj, "image-ap")
        table += [str(np.round(image_auroc * 100, 1)), str(np.round(image_ap * 100, 1))]
        image_auroc_list.append(image_auroc)
        image_ap_list.append(image_ap)
        table_ls.append(table)

    table_ls.append(["mean", str(np.round(np.mean(image_auroc_list) * 100, 1)), str(np.round(np.mean(image_ap_list) * 100, 1))])
    return tabulate(table_ls, headers=["objects", "image_auroc", "image_ap"], tablefmt="pipe")


def test_stage1_image(args):
    """Stage 1 inference: Evaluate image score using fused CLIP+DINO."""
    os.makedirs(args.save_path, exist_ok=True)
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

    # Load Stage 1 checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    model_clip.to("cpu")
    prompt_learner = AnomalyCLIP_PromptLearner(model_clip, clip_cfg)
    prompt_learner.load_state_dict(checkpoint["stage1_prompt_learner"])
    model_clip.to(device)
    prompt_learner.to(device).eval()

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
    detail_model.load_state_dict(checkpoint["detail_model"])
    detail_model.eval()

    # NaiveFusionHead (no learnable params, no state_dict to load)
    fusion_head = NaiveFusionHead().to(device)
    fusion_head.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list
    results = {obj: {"gt_sp": [], "pr_sp": [], "imgs_masks": [], "anomaly_maps": []} for obj in obj_list}

    text_features = encode_text_features(model_clip, prompt_learner, device)

    for items in tqdm(test_loader):
        image = items["img"].to(device)
        cls_name = items["cls_name"]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]["imgs_masks"].append(gt_mask)
        results[cls_name[0]]["gt_sp"].extend(items["anomaly"].detach().cpu())

        with torch.no_grad():
            # Stage 1: image score from fused CLIP+DINO
            image_features, _ = model_clip.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = F.normalize(image_features, dim=-1)
            clip_image_logits = (image_features.unsqueeze(1) @ text_features.permute(0, 2, 1))[:, 0, ...] / 0.07

            _, dino_image_logits, _ = detail_model(image=image, image_size=args.image_size)

            fused_image_logits = fusion_head(clip_image_logits, dino_image_logits)
            image_score = torch.softmax(fused_image_logits, dim=-1)[:, 1]

            results[cls_name[0]]["pr_sp"].extend(image_score.detach().cpu())
            
            # Dummy map since we only evaluate image-level
            dummy_map = torch.zeros((1, args.image_size, args.image_size), dtype=torch.float32)
            results[cls_name[0]]["anomaly_maps"].append(dummy_map)

    table = evaluate_image_only(results, obj_list)
    logger.info("\n%s", table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AnomalyCLIP Stage 1 Evaluation")
    parser.add_argument("--data_path", type=str, default="", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default="", help="path to save results")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to stage 1 checkpoint")
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--seed", type=int, default=111, help="random seed")

    parser.add_argument("--dino_model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--dino_source", type=str, default="local", choices=["local", "official", "timm", "torchhub", "auto"])
    parser.add_argument("--dino_pretrained", action="store_true")
    parser.add_argument("--dino_ckpt_path", type=str, default="")
    parser.add_argument("--dino_repo_or_dir", type=str, default="facebookresearch/dinov3")
    parser.add_argument("--dino_repo_source", type=str, default="github", choices=["github", "local"])
    parser.add_argument("--detail_num_queries", type=int, default=2)
    parser.add_argument("--detail_attn_heads", type=int, default=8)
    parser.add_argument("--detail_dropout", type=float, default=0.1)

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test_stage1_image(args)
