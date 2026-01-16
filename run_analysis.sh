#!/bin/bash

mi_threshold=( 0.25 )
min_pairs=( 2 )
dataset="webnlg"
lambda_sem=( 0.3 0.5 0.7 1.0 1.5 2.0 )
lambda_type=( 0.3 0.5 0.7 1.0 1.5 2.0 )
lambda_new=( 0.5 1.0 1.5 2.0 2.5 3.0 5.0)

for mi in "${mi_threshold[@]}"; do
    for min_pair in "${min_pairs[@]}"; do
        for sem in "${lambda_sem[@]}"; do
            for type in "${lambda_type[@]}"; do
                for new in "${lambda_new[@]}"; do
                    python distortion_analysis.py --mi_threshold $mi --min_pairs $min_pair --dataset $dataset --lambda_sem $sem --lambda_type $type --lambda_new $new | tee output/$dataset/sem_${sem}_type_${type}_new_${new}.log
                done
            done
        done
    done
done
