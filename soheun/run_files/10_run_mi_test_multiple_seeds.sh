for i in {0..4}
do
	j=$(($i + 1))
	sed -e "s/SEED_START=00/SEED_START=${i}0/g" -e "s/SEED_END=10/SEED_END=${j}0/g" run_mi_test.sh > "run_mi_test_$i.sh"
	chmod +x "run_mi_test_$i.sh"
	sbatch "run_mi_test_$i.sh"
	rm -f "run_mi_test_$i.sh"
	sleep 1
done
