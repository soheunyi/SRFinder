for seed in 42 52
	do
	for sr in 0.0025 0.005 0.0075 0.01
	do
		python fvt_first_training.py --seed $seed --signal_ratio $sr --n_3b 2000000 --n_all4b 2000000
	done
done