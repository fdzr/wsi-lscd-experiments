#!/bin/bash


methods="chinese_whispers correlation_clustering wsbm spectral_clustering"
datasets="dwug_data_annotated_only dwug_new_data_annotated_only dwug_old_data_annotated_only"

number_of_experiments=5

for index $(seq 1 $number_of_experiments)
do
    for method in $methods 
    do
        mkdir -p ../outputs/experiment-results/results_$index/$method/cache
        mkdir -p ../outputs/experiment-results/results_$index/$method/results
        mkdir -p ../outputs/output_freq_clusters/run_$index
    done
done

for d in $datasets
do
    mkdir -p ../outputs/avg_predicted_clusters/$d/
done

for m in $methods
do
    mkdir -p ../logs/$m
    touch ../logs/$m/logs.txt
done

mkdir -p ../outputs/outputs_spearman/
mkdir -p ../outputs/results_whole_dataset/
mkdir -p ../outputs/results_old_dataset/
mkdir -p ../outputs/results_new_dataset/

python3 generate_cache_file.py -n $number_of_experiments

jupyter nbconvert --to notebook --execute Chinese_Whispers.ipynb & 
jupyter nbconvert --to notebook --execute Correlation_clustering.ipynb & 
jupyter nbconvert --to notebook --execute WSBM.ipynb & 
jupyter nbconvert --to notebook --execute Spectral_clustering.ipynb &

wait

python3 compute_results.py

jupyter nbconvert --to notebook --execute 5-fold-cv/5-fold-cross-validation.ipynb