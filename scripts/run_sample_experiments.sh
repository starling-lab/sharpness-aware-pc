#!/usr/bin/env bash
# author:  Sahil Sidheekh
# date:    2025-07-13
# ------------------------------------------------------------------
# Example script to run Sharpness-Aware Learning experiments 
# ------------------------------------------------------------------

# ------------------------------------------------------------------
#  EinsumNets over Synthetic Datasets
# ------------------------------------------------------------------
n_samples=0.01
model_name=EinsumNet
leaf_distribution=NormalArray
num_sums=10
num_input_distributions=10
graph_type=ratspn
num_repetitions=10
lr=1e-1

# Example: Spiral Dataset with 2 variables
dataset=spiral
num_vars=2

# Unregularized Baseline (Set trainer.mu=0)
mu_strength=0.0
echo "## Running Unregularized Baseline EinsumNet on $dataset dataset"
python -m hessian_reg.run \
  dataset.name=$dataset dataset.n_samples=$n_samples \
  model.backend=pfc model.model_name=$model_name model.num_vars=$num_vars \
  model.leaf_distribution=$leaf_distribution model.num_sums=$num_sums \
  model.num_input_distributions=$num_input_distributions model.graph_type=$graph_type \
  +model.num_repetitions=$num_repetitions +model.leaf_config.mean=0 +model.leaf_config.std=1 \
  trainer.mode=sgd trainer.mu=$mu_strength trainer.lambda_=1.0 trainer.lr=$lr trainer.save_final=False\
  seed=0 

# Sharpness-Aware Learning (Set trainer.mu to any float value to see corresponding effect.)
mu_strength=auto # Using adaptive schedule for μ 
echo "## Running Sharpness-Aware Learning with EinsumNet on $dataset dataset"
python -m hessian_reg.run \
  dataset.name=$dataset dataset.n_samples=$n_samples \
  model.backend=pfc model.model_name=$model_name model.num_vars=$num_vars \
  model.leaf_distribution=$leaf_distribution model.num_sums=$num_sums \
  model.num_input_distributions=$num_input_distributions model.graph_type=$graph_type \
  +model.num_repetitions=$num_repetitions +model.leaf_config.mean=0 +model.leaf_config.std=1 \
  trainer.mode=sgd trainer.mu=$mu_strength trainer.lambda_=1.0 trainer.lr=$lr trainer.save_final=False\
  seed=0 

# Note: Adjust hyperparameters and settings as needed to run on other datasets.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Pyjuice HCLT Model on the binary density estimation datasets (DEBD)
# ------------------------------------------------------------------

batch_size=200
epochs=100
lr=0.1
lambda_=1
model_name=TensorCircuit
backend=pyjuice
leaf_distribution=categorical
K=100
repetitions=10
n_samples=0.01
graph_type=hclt
mode=em

# Set number of variables and depth based on dataset
dataset=jester 
num_vars=100
depth=6

# Unregularized Baseline (Set trainer.mu=0)
mu_strength=0.0
echo "## Running Unregularized Baseline PyJuice HCLT on $dataset dataset"
python -m hessian_reg.run \
    dataset.n_samples=$n_samples \
    dataset.name=$dataset dataset.batch_size=$batch_size \
    trainer.mu=$mu_strength trainer.lambda_=$lambda_ trainer.epochs=$epochs trainer.mode=$mode trainer.lr=$lr trainer.loss_agg=mean \
    model.num_vars=$num_vars model.backend=$backend model.model_name=$model_name \
    model.leaf_distribution=$leaf_distribution  +model.num_latents=$K model.depth=$depth \
    model.num_input_distributions=$K model.graph_type=$graph_type \
    +model.num_repetitions=$repetitions  \
    +model.leaf_config.num_cats=2 \
    seed=0 

# Sharpness-Aware Learning (Set trainer.mu to any float value to see corresponding effect.)
mu_strength=0.1
echo "## Running Sharpness-Aware Learning with PyJuice HCLT on $dataset dataset with μ=$mu_strength"
python -m hessian_reg.run \
    dataset.n_samples=$n_samples \
    dataset.name=$dataset dataset.batch_size=$batch_size \
    trainer.mu=$mu_strength trainer.lambda_=$lambda_ trainer.epochs=$epochs trainer.mode=$mode trainer.lr=$lr trainer.loss_agg=mean \
    model.num_vars=$num_vars model.backend=$backend model.model_name=$model_name \
    model.leaf_distribution=$leaf_distribution  +model.num_latents=$K model.depth=$depth \
    model.num_input_distributions=$K model.graph_type=$graph_type \
    +model.num_repetitions=$repetitions  \
    +model.leaf_config.num_cats=2 \
    seed=0 

