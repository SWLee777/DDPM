#!/usr/bin/env bash
# =============================================================
# F-experiment: Two-stage test across multiple datasets
# =============================================================
set -e

device=0
stage1_ckpt=./checkpoint/stage1_visa/epoch_15.pth
stage2_ckpt=./checkpoint/stage2_visa/epoch_15.pth

dino_weight=./weight/dinov3_vitb16_pretrain_lvd1689m.pth
dino_repo_or_dir=facebookresearch/dinov3
dino_repo_source=github

for dataset in visa mvtec; do
  data_path=./data/${dataset}
  save_path=./results/two_stage_${dataset}/
  mkdir -p ${save_path}

  echo "Testing on ${dataset} ..."

  CUDA_VISIBLE_DEVICES=${device} python test.py \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --save_path ${save_path} \
    --stage1_ckpt ${stage1_ckpt} \
    --checkpoint_path ${stage2_ckpt} \
    --features_list 6 12 18 24 \
    --feature_map_layer 0 1 2 3 \
    --image_size 518 \
    --depth 9 \
    --n_ctx 12 \
    --t_n_ctx 4 \
    --sigma 4 \
    --dino_model_name dinov3_vitb16 \
    --dino_source local \
    --dino_ckpt_path ${dino_weight} \
    --dino_repo_or_dir ${dino_repo_or_dir} \
    --dino_repo_source ${dino_repo_source} \
    --detail_num_queries 2
done
