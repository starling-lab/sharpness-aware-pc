#!/usr/bin/env bash
# author:  Sahil Sidheekh
# date:    2025-07-13
# ------------------------------------------------------------------

for dataset in spiral pinwheel two_moons; do
    for data_frac in 0.01 0.05 0.1 0.5 1.0; do
        mkdir -p figures/2d/$dataset
        JS=results/aaai26/2d/sharpness_vs_nsamples/pfc/$dataset/EinsumNet-K10/trial-*/SGD/R0/performance_summary_*.json
        python hessian_reg/utils/plot_learning.py $JS \
            --title $dataset"· SGD · μ=0" \
            --png figures/2d/$dataset/"EinsumNet-K10_sgd_R0_N$data_frac.png" \
            --pdf figures/2d/$dataset/"EinsumNet-K10_sgd_R0_N$data_frac.pdf" \
            --data_frac $data_frac

        JS=results/aaai26/2d/sharpness_vs_nsamples/pfc/$dataset/EinsumNet-K10/trial-*/SGD/Rauto/performance_summary_*.json
        python hessian_reg/utils/plot_learning.py $JS \
            --title $dataset"· SGD · μ>0" \
            --png figures/2d/$dataset/"EinsumNet-K10_sgd_Rauto_N$data_frac.png" \
            --pdf figures/2d/$dataset/"EinsumNet-K10_sgd_Rauto_N$data_frac.pdf" \
            --data_frac $data_frac
    done
done

for dataset in helix knotted interlocked_circles twisted_eight; do
    for data_frac in 0.01 0.05 0.1 0.5 1.0; do
        mkdir -p figures/3d/$dataset
        JS=results/aaai26/3d/sharpness_vs_nsamples/pfc/$dataset/EinsumNet-K10/trial-*/SGD/R0/performance_summary_*.json
        python hessian_reg/utils/plot_learning.py $JS \
            --title $dataset"· SGD · μ=0" \
            --png figures/3d/$dataset/"EinsumNet-K10_sgd_R0_N$data_frac.png" \
            --pdf figures/3d/$dataset/"EinsumNet-K10_sgd_R0_N$data_frac.pdf" \
            --data_frac $data_frac

        JS=results/aaai26/3d/sharpness_vs_nsamples/pfc/$dataset/EinsumNet-K10/trial-*/SGD/Rauto/performance_summary_*.json
        python hessian_reg/utils/plot_learning.py $JS \
            --title $dataset"· SGD · μ>0" \
            --png figures/3d/$dataset/"EinsumNet-K10_sgd_Rauto_N$data_frac.png" \
            --pdf figures/3d/$dataset/"EinsumNet-K10_sgd_Rauto_N$data_frac.pdf" \
            --data_frac $data_frac
    done
done