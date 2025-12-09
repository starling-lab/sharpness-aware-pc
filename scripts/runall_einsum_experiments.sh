#!/usr/bin/env bash
# author:  Sahil Sidheekh
# date:    2025-07-13
# ------------------------------------------------------------------
# Sharpness vs n‑samples — synthetic datasets with EinsumNets
# Script to run all experiments for 2D and 3D synthetic datasets in parallel,
# assigning jobs to 4 GPUs (assuming each has 24GB memory) in a round-robin fashion.
# ------------------------------------------------------------------

set -euo pipefail
# --- dataset groups ----------------------------------------------------------
synth_datasets_2d=(spiral pinwheel two_moons)
synth_datasets_3d=(helix knotted bent_lissajous twisted_eight interlocked_circles)
save_final=1

# --- resolve what to run -----------------------------------------------------
datasets_to_run=()

for arg in "$@"; do
    case $arg in
        synth2d) datasets_to_run+=( "${synth_datasets_2d[@]}" ) ;;
        synth3d) datasets_to_run+=( "${synth_datasets_3d[@]}" ) ;;
        *)       datasets_to_run+=( "$arg" ) ;;             # explicit dataset
    esac
done

# If no CLI args ⇒ run *everything*
if (( ${#datasets_to_run[@]} == 0 )); then
    datasets_to_run=(
        "${synth_datasets_2d[@]}"
        "${synth_datasets_3d[@]}"
    )
fi

# --- remove duplicates --------------------------------------------
if (( ${#datasets_to_run[@]} > 1 )); then
    declare -A seen
    uniq=()
    for ds in "${datasets_to_run[@]}"; do
        if [[ -z ${seen[$ds]+_} ]]; then
            uniq+=( "$ds" )
            seen[$ds]=1
        fi
    done
    datasets_to_run=( "${uniq[@]}" )
fi

# --- run ---------------------------------------------------------------------
printf 'Running experiments for datasets: %s\n' "${datasets_to_run[*]}"

to_run(){ [[ " ${datasets_to_run[*]} " == *" $1 "* ]]; }

# ─────────────────────────── globals ────────────────────────────────#
graph_type=ratspn
repetitions=10
lr=0.1
epochs=200
lambda_=1
trials=(0 1 2 3 4)
# ────────────────────────── GPU helpers ─────────────────────────────#
# Map (mode, λ) → gpu id
choose_gpu() {                       # $1=mode  $2=mu_strength
    case "$1/$2" in
        sgd/0)  echo 2 ;;
        em/0)   echo 1 ;;
        sgd/*)  echo 3 ;;
        *)      echo 3 ;;
    esac
}
next_gpu=0
# =========== 1) Synthetic: allow max 2 *datasets* / GPU =============#
backend=pfc
leaf_distribution=NormalArray
model_name=EinsumNet
K=10
batch_size=200

# per‑GPU PID queues (strings of pids) --------------------------------
declare -a queue                 # queue[0] .. queue[3]

launch_synth() {                 # $1=gpu  rest=command …
    local gpu=$1; shift
    while :; do
        # prune finished pids from queue[gpu]
        local alive=()
        for p in ${queue[$gpu]:-}; do
            if kill -0 "$p" 2>/dev/null; then alive+=("$p"); fi
        done
        queue[$gpu]="${alive[*]}"
        if (( ${#alive[@]} < 24 )); then break; fi   # slot available
        wait "${alive[0]}"                            # wait oldest pid
    done
    CUDA_VISIBLE_DEVICES=$gpu "$@" &
    queue[$gpu]="${queue[$gpu]:-} $!"                 # append new pid
}


num_vars=2
base_dir=results/aaai26/2d

for trial in "${trials[@]}"; do 
for dataset in "${synth_datasets_2d[@]}"; do

  if ! to_run "$dataset"; then
    printf '❌  %s is NOT scheduled to run.\n' "$dataset"
    continue
  fi

  exp_name="sharpness_vs_nsamples-${backend}-${dataset}-${model_name}-K${K}-trial${trial}"
  run_dir="${base_dir}/sharpness_vs_nsamples/${backend}/${dataset}/${model_name}-K${K}/trial-${trial}"
  mkdir -p console/2d/$dataset/${backend}

  for mode in sgd; do
    for mu_strength in  0 0.01 0.1 0.5 1.0 auto; do
      gpu=$next_gpu
      next_gpu=$(( (next_gpu + 1) % 4 ))
      exp_tag="${exp_name}-${mode^^}-R${mu_strength}"
      subdir="${run_dir}/${mode^^}/R${mu_strength}"

      echo -n "" > console/2d/$dataset/${backend}/${exp_name}-${mode}-reg-${mu_strength}-trial-${trial}.log

      launch_synth "$gpu" \
      python -m hessian_reg.experiments.sharpness \
        dataset.name=$dataset dataset.batch_size=$batch_size \
        trainer.mu=$mu_strength trainer.lambda_=$lambda_ \
        trainer.epochs=$epochs trainer.mode=$mode trainer.lr=$lr \
        trainer.save_final=$save_final \
        model.num_vars=$num_vars model.backend=$backend model.model_name=$model_name \
        model.leaf_distribution=$leaf_distribution model.num_sums=$K \
        model.num_input_distributions=$K model.graph_type=$graph_type \
        +model.num_repetitions=$repetitions +model.leaf_config.mean=0 +model.leaf_config.std=1 \
        ++seed=$trial \
        exp.name=$exp_tag exp.base_dir=$base_dir hydra.run.dir=$subdir >> console/2d/$dataset/${backend}/${exp_name}-${mode}-reg-${mu_strength}-trial-${trial}.log 
    done
  done
  echo "[SYNTH] $dataset queued."
done
done

num_vars=3
base_dir=results/aaai26/3d

for trial in "${trials[@]}"; do
for dataset in "${synth_datasets_3d[@]}"; do
  if ! to_run "$dataset"; then
    printf '❌  %s is NOT scheduled to run.\n' "$dataset"
    continue
  fi

  exp_name="sharpness_vs_nsamples-${backend}-${dataset}-${model_name}-K${K}-trial${trial}"
  run_dir="${base_dir}/sharpness_vs_nsamples/${backend}/${dataset}/${model_name}-K${K}/trial-${trial}"
  mkdir -p console/3d/$dataset/${backend}
  for mode in sgd; do
    for mu_strength in 0 0.01 0.1 0.5 1.0 auto; do
      gpu=$next_gpu
      next_gpu=$(( (next_gpu + 1) % 4 ))
      exp_tag="${exp_name}-${mode^^}-R${mu_strength}"
      subdir="${run_dir}/${mode^^}/R${mu_strength}"

      echo -n "" > console/3d/$dataset/${backend}/${exp_name}-${mode}-reg-${mu_strength}-trial-${trial}.log

      launch_synth "$gpu" \
      python -m hessian_reg.experiments.sharpness \
        trainer.mu=$mu_strength trainer.lambda_=$lambda_ trainer.epochs=$epochs trainer.mode=$mode trainer.lr=$lr \
        dataset.name=$dataset dataset.batch_size=$batch_size \
        model.num_vars=$num_vars model.backend=$backend model.model_name=$model_name \
        model.leaf_distribution=$leaf_distribution model.num_sums=$K \
        model.num_input_distributions=$K model.graph_type=$graph_type \
        +model.num_repetitions=$repetitions +model.leaf_config.mean=0 +model.leaf_config.std=1 \
        ++seed=$trial \
        exp.name=$exp_tag exp.base_dir=$base_dir hydra.run.dir=$subdir >> console/3d/$dataset/${backend}/${exp_name}-${mode}-reg-${mu_strength}-trial-${trial}.log 
    done
  done
  echo "[SYNTH] $dataset queued."
done
done 

# # ────────────────── wait for synthetic queues to settle ─────────────#
for gpu_q in ${queue[*]:-}; do
    for pid in $gpu_q; do wait "$pid"; done
done

# ------------------------------------------------------------------
# 2) Plot learning curves + Visualize loss surfaces
# ------------------------------------------------------------------
echo "Plotting learning curves and saving them to: figures/ ..."
bash scripts/plot_learning_curves.sh

echo "Visualizing Loss Surfaces ..."
bash scripts/visualize_loss_surface.sh

echo "All jobs finished."
# ------------------------------------------------------------------
