#!/bin/bash

mi_threshold=[ 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 ]
min_pairs=[ 2 ]
dataset="webnlg"

for mi in "${mi_threshold[@]}"; do
    for min_pair in "${min_pairs[@]}"; do
        python mi_threshold_analysis.py --mi_threshold $mi --min_pairs $min_pair --dataset $dataset | tee output/$dataset/mi_${mi}_minpairs_${min_pair}.log
    done
done

