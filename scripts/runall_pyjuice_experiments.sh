#!/usr/bin/env bash
# author:  Sahil Sidheekh
# date:    2025-07-13
# ------------------------------------------------------------------
# Sharpness vs n‑samples — ALL DEBD datasets using PyJuice HCLTs in parallel
# assigning jobs to 4 GPUs (assuming each has 24GB memory) in a round-robin fashion.
# ------------------------------------------------------------------

set -euo pipefail
mkdir -p console 

# --- dataset groups ----------------------------------------------------------
debd_datasets=( dna cwebkb tmovie cr52 book jester c20ng pumsb_star accidents baudio bnetflix nltcs plants tretail msweb kosarek kdd msnbc ad bbc )

# --- resolve what to run -----------------------------------------------------
datasets_to_run=()

for arg in "$@"; do
    case $arg in
        debd)    datasets_to_run+=( "${debd_datasets[@]}" ) ;;
        *)       datasets_to_run+=( "$arg" ) ;;             # explicit dataset
    esac
done

# If no CLI args ⇒ run *everything*
if (( ${#datasets_to_run[@]} == 0 )); then
    datasets_to_run=(
        "${debd_datasets[@]}"
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

launch_debd() {                 # $1=gpu  rest=command …
    local gpu=$1; shift
    while :; do
        # prune finished pids from queue[gpu]
        local alive=()
        for p in ${queue[$gpu]:-}; do
            if kill -0 "$p" 2>/dev/null; then alive+=("$p"); fi
        done
        queue[$gpu]="${alive[*]}"
        if (( ${#alive[@]} < 20 )); then break; fi   # slot available
        wait "${alive[0]}"                            # wait oldest pid
    done
    CUDA_VISIBLE_DEVICES=$gpu "$@" &
    queue[$gpu]="${queue[$gpu]:-} $!"               # append new pid
}

# # =========== 2) DEBD binary: 1 job per GPU (reuse launcher) =========#
backend=pyjuice
leaf_distribution=categorical
model_name=TensorCircuit
graph_type=hclt
repetitions=10
lambda_=1
K=100
batch_size=200
epochs=100
lr=0.1
base_dir=results/aaai26/debd
# reuse queue[] + launch() from previous answer ----------------------
declare -a pids
launch() {
    local gpu=$1; shift
    if [[ -n "${pids[$gpu]:-}" ]]; then wait "${pids[$gpu]}"; fi
    CUDA_VISIBLE_DEVICES=$gpu "$@" &
    pids[$gpu]=$!
}

# ------------- round-robin GPU allocator --------------------------
next_gpu=0
pick_gpu() {                        # returns 0,1,2,3 in a cycle
    echo $next_gpu
    next_gpu=$(( (next_gpu + 1) % 4 ))
}

trials=(1 2 3 4 5)
pseudocount=0.0
for trial in "${trials[@]}"
do 
while read -r dataset num_vars depth; do
  [[ -z $dataset ]] && continue

  if ! to_run "$dataset"; then
    printf '❌  %s is NOT scheduled to run.\n' "$dataset"
    continue
  fi
  exp_name="sharpness_vs_nsamples-${backend}-${dataset}-${model_name}-K${K}-trial${trial}"
  run_dir="${base_dir}/sharpness_vs_nsamples/${backend}/${dataset}/${model_name}-K${K}/trial-${trial}"
  
  mkdir -p console/debd/$dataset/pyjuice
  for mode in em; do
    for mu_strength in 0 0.01 0.05 0.1 0.5 1.0; do
      gpu=$next_gpu
      next_gpu=$(( (next_gpu + 1) % 4 ))

      
      exp_tag="${exp_name}-${mode^^}-R${mu_strength}"
      subdir="${run_dir}/${mode^^}/R${mu_strength}"

      echo "Launching: $exp_tag on GPU $gpu"
      # launch the job on the chosen GPU
      launch_debd "$gpu" \
        python -m hessian_reg.experiments.sharpness \
        trainer.mu=$mu_strength trainer.lambda_=$lambda_ trainer.epochs=$epochs trainer.mode=$mode trainer.lr=$lr trainer.loss_agg=mean \
        dataset.name=$dataset dataset.batch_size=$batch_size \
        model.num_vars=$num_vars model.backend=$backend model.model_name=$model_name \
        model.leaf_distribution=$leaf_distribution  +model.num_latents=$K model.depth=$depth \
        model.num_input_distributions=$K model.graph_type=$graph_type \
        +model.num_repetitions=$repetitions  \
        +model.leaf_config.num_cats=2 \
        ++seed=$trial \
        trainer.pseudocount=$pseudocount \
        exp.name=$exp_tag exp.base_dir=$base_dir hydra.run.dir=$subdir >> console/debd/${dataset}/pyjuice/${exp_name}-Reg-${mu_strength}-${mode}.log
    done
  done
  echo "[DEBD] $dataset queued."
done << 'EOF'
dna         180   7
cwebkb      839   9
tmovie      500   8
cr52        889   9
book        500   8
jester      100   6
c20ng       910   9
pumsb_star  163   7
accidents   111   6
baudio      100   6
bnetflix    100   6
nltcs        16   4
plants       69   6
tretail     135   7
msweb       294   8
kosarek     190   7
kdd          64   6
msnbc        17   4
ad          1556  10
bbc         1058  10
voting      1359  10
moviereview 1001  9
EOF
done

wait for last DEBD jobs
wait "${pids[@]:-}"

echo " All jobs finished."
