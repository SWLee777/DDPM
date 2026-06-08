#!/usr/bin/env bash
# =============================================================
# F-experiment: Two-stage training pipeline
#   Stage 1 → CLIP + DINO image-level (NaiveFusionHead)
#   Stage 2 → CLIP patches + SpatialPriorFiLMModulator
# =============================================================
set -e

device=0
dataset=visa
data_path=./data/${dataset}

dino_weight=./weight/dinov3_vitb16_pretrain_lvd1689m.pth
dino_repo_or_dir=facebookresearch/dinov3
dino_repo_source=github

stage1_dir=./checkpoint/stage1_${dataset}/
stage2_dir=./checkpoint/stage2_${dataset}/

# =====================  Stage 1  =====================
echo "============================================"
echo "  Stage 1: CLIP + DINO image-level training"
echo "============================================"
mkdir -p ${stage1_dir}

CUDA_VISIBLE_DEVICES=${device} python train.py \
  --training_stage stage1 \
  --dataset ${dataset} \
  --train_data_path ${data_path} \
  --save_path ${stage1_dir} \
  --features_list 6 12 18 24 \
  --feature_map_layer 0 1 2 3 \
  --image_size 518 \
  --batch_size 8 \
  --epoch 15 \
  --save_freq 1 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --learning_rate 0.001 \
  --dino_model_name dinov3_vitb16 \
  --dino_source local \
  --dino_ckpt_path ${dino_weight} \
  --dino_repo_or_dir ${dino_repo_or_dir} \
  --dino_repo_source ${dino_repo_source} \
  --detail_num_queries 2 \
  --detail_attn_heads 8 \
  --detail_dropout 0.1 \
  --stage1_clip_weight 1.0 \
  --stage1_dino_weight 1.0 \
  --stage1_fused_weight 1.0

# =====================  Stage 2  =====================
echo "============================================"
echo "  Stage 2: Pixel-level + FiLM modulation"
echo "============================================"
stage1_ckpt=${stage1_dir}/epoch_15.pth
mkdir -p ${stage2_dir}

CUDA_VISIBLE_DEVICES=${device} python train.py \
  --training_stage stage2 \
  --dataset ${dataset} \
  --train_data_path ${data_path} \
  --save_path ${stage2_dir} \
  --stage1_ckpt ${stage1_ckpt} \
  --features_list 6 12 18 24 \
  --feature_map_layer 0 1 2 3 \
  --image_size 518 \
  --batch_size 8 \
  --epoch 15 \
  --save_freq 1 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --learning_rate 0.001 \
  --dino_model_name dinov3_vitb16 \
  --dino_source local \
  --dino_ckpt_path ${dino_weight} \
  --dino_repo_or_dir ${dino_repo_or_dir} \
  --dino_repo_source ${dino_repo_source} \
  --detail_num_queries 2 \
  --detail_attn_heads 8 \
  --detail_dropout 0.1 \
  --prior_dropout 0.3 \
  --stage2_aux_img_weight 0.1

echo "============================================"
echo "  Training complete!"
echo "  Stage1 ckpt: ${stage1_dir}"
echo "  Stage2 ckpt: ${stage2_dir}"
echo "============================================"
