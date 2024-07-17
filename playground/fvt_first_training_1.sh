for sr in 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
do
	for seed in 42 52 62
	do
		python fvt_first_training.py --seed $seed --signal_ratio $sr 
	done
done