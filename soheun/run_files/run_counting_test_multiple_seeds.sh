for i in {4..4}
do
	j=$(($i + 1))
	sed -e "s/SEED_START=0/SEED_START=${i}0/g" -e "s/SEED_END=10/SEED_END=${j}0/g" run_counting_test.sh > "run_counting_test_$i.sh"
	chmod +x "run_counting_test_$i.sh"
	sbatch "run_counting_test_$i.sh"
	rm -f "run_counting_test_$i.sh"
done