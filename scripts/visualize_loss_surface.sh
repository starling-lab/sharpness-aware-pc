#!/usr/bin/env bash
# author:  Sahil Sidheekh
# date:    2025-07-13
# ------------------------------------------------------------------

base_dir=results/aaai26/2d
backend=pfc
leaf_distribution=NormalArray
model_name=EinsumNet
K=10
batch_size=200

# ─────────────────────────── globals ────────────────────────────────#
num_vars=2
graph_type=ratspn
repetitions=10
lr=0.1
epochs=200
lambda_norm=1
trials=(0 1 2 3 4)
synth_datasets_2d=(spiral pinwheel two_moons)
next_gpu=0

for trial in "${trials[@]}"; do 
for n_sample in 0.01 0.05 0.1 0.5 1.0; do
for dataset in "${synth_datasets_2d[@]}"; do
  exp_name="sharpness_vs_nsamples-${backend}-${dataset}-${model_name}-K${K}-trial${trial}"
  run_dir="${base_dir}/sharpness_vs_nsamples/${backend}/${dataset}/${model_name}-K${K}/trial-${trial}"

  for mode in sgd; do
    for mu_strength in 0 auto; do
      gpu=$next_gpu
      next_gpu=$(( (next_gpu + 1) % 4 ))
      exp_tag="${exp_name}-${mode^^}-R${mu_strength}"
      subdir="${run_dir}/${mode^^}/R${mu_strength}"

      python -m hessian_reg.experiments.loss_landscape \
        dataset.name=$dataset dataset.batch_size=$batch_size \
        trainer.mu=$mu_strength trainer.lambda_=$lambda_norm \
        trainer.epochs=$epochs trainer.mode=$mode trainer.lr=$lr \
        model.num_vars=$num_vars model.backend=$backend model.model_name=$model_name \
        model.leaf_distribution=$leaf_distribution model.num_sums=$K \
        model.num_input_distributions=$K model.graph_type=$graph_type \
        +model.num_repetitions=$repetitions +model.leaf_config.mean=0 +model.leaf_config.std=1 \
        dataset.n_samples=$n_sample \
        ++seed=$trial \
        exp.name=$exp_tag exp.base_dir=$base_dir hydra.run.dir=$subdir 
    done
  done
  echo "[SYNTH] $dataset queued."
done
done
done




num_vars=3
graph_type=ratspn
repetitions=10
lr=0.1
epochs=200
lambda_norm=1

trials=(0 1 2 3 4)
synth_datasets_3d=(helix knotted bent_lissajous twisted_eight interlocked_circles)
n_sample=0.01

for trial in "${trials[@]}"; do 
for n_sample in 0.01 0.05 0.1 0.5 1.0; do
for dataset in "${synth_datasets_3d[@]}"; do
  exp_name="sharpness_vs_nsamples-${backend}-${dataset}-${model_name}-K${K}-trial${trial}"
  run_dir="${base_dir}/sharpness_vs_nsamples/${backend}/${dataset}/${model_name}-K${K}/trial-${trial}"

  for mode in sgd; do
    for mu_strength in 0 auto; do
      gpu=$next_gpu
      next_gpu=$(( (next_gpu + 1) % 4 ))
      exp_tag="${exp_name}-${mode^^}-R${mu_strength}"
      subdir="${run_dir}/${mode^^}/R${mu_strength}"

      python -m hessian_reg.experiments.loss_landscape \
        dataset.name=$dataset dataset.batch_size=$batch_size \
        trainer.mu=$mu_strength trainer.lambda_=$lambda_norm \
        trainer.epochs=$epochs trainer.mode=$mode trainer.lr=$lr \
        model.num_vars=$num_vars model.backend=$backend model.model_name=$model_name \
        model.leaf_distribution=$leaf_distribution model.num_sums=$K \
        model.num_input_distributions=$K model.graph_type=$graph_type \
        +model.num_repetitions=$repetitions +model.leaf_config.mean=0 +model.leaf_config.std=1 \
        dataset.n_samples=$n_sample \
        +seed=$trial \
        exp.name=$exp_tag exp.base_dir=$base_dir hydra.run.dir=$subdir 
    done
  done
  echo "[SYNTH] $dataset queued."
done
done
done