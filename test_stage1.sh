#!/usr/bin/env bash
# =============================================================
# F-experiment: Evaluate Stage 1 checkpoints (image-level only)
#   Loops through all epoch checkpoints to find the best one.
# =============================================================
set -e

device=0
dataset=visa
data_path=./data/${dataset}
stage1_dir=./checkpoint/stage1_${dataset}/
save_path=./results/stage1_tuning_${dataset}/

dino_weight=./weight/dinov3_vitb16_pretrain_lvd1689m.pth
dino_repo_or_dir=facebookresearch/dinov3
dino_repo_source=github

mkdir -p ${save_path}

echo "============================================"
echo "  Evaluating Stage 1 image-level checkpoints"
echo "============================================"

for epoch in $(seq 1 15); do
  ckpt_path="${stage1_dir}/epoch_${epoch}.pth"
  if [ -f "$ckpt_path" ]; then
    echo "Evaluating Epoch ${epoch} ..."
    CUDA_VISIBLE_DEVICES=${device} python test_stage1_image.py \
      --dataset ${dataset} \
      --data_path ${data_path} \
      --save_path ${save_path} \
      --checkpoint_path ${ckpt_path} \
      --features_list 6 12 18 24 \
      --image_size 518 \
      --depth 9 \
      --n_ctx 12 \
      --t_n_ctx 4 \
      --dino_model_name dinov3_vitb16 \
      --dino_source local \
      --dino_ckpt_path ${dino_weight} \
      --dino_repo_or_dir ${dino_repo_or_dir} \
      --dino_repo_source ${dino_repo_source} \
      --detail_num_queries 2
  fi
done

echo "============================================"
echo "  Evaluation complete!"
echo "  Check ${save_path} for the best epoch."
echo "  Use that checkpoint as --stage1_ckpt for Stage 2."
echo "============================================"
