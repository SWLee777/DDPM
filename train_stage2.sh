#!/usr/bin/env bash
# =============================================================
# F-experiment Stage 2: Pixel-level + SpatialPriorFiLMModulator
# =============================================================
set -e

device=0
dataset=visa
data_path=./data/${dataset}
save_dir=./checkpoint/stage2_${dataset}/
stage1_ckpt=./checkpoint/stage1_${dataset}/epoch_15.pth

dino_weight=./weight/dinov3_vitb16_pretrain_lvd1689m.pth
dino_repo_or_dir=facebookresearch/dinov3
dino_repo_source=github

mkdir -p ${save_dir}

CUDA_VISIBLE_DEVICES=${device} python train.py \
  --training_stage stage2 \
  --dataset ${dataset} \
  --train_data_path ${data_path} \
  --save_path ${save_dir} \
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
