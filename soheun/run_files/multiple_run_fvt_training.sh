for sr in {0.01,}
do
	for ns in {3.0,}
	do
		CONFIG_FILENAME="tmp_smeared_fvt_training_${sr}_${ns}.yml"
		RUN_FILENAME="run_fvt_training_${sr}_${ns}.sh"

		sed -e "s/experiment_name: smeared_fvt_training/experiment_name: smeared_fvt_training_noise_scale/g" "../configs/smeared_fvt_training.yml" > "../configs/$CONFIG_FILENAME"
		sed -i "s/noise_scale: 1.0/noise_scale: ${ns}/g" "../configs/$CONFIG_FILENAME"
		sed -e "s/CONFIG_FILENAME=\"smeared_fvt_training_small.yml\"/CONFIG_FILENAME=\"${CONFIG_FILENAME}\"/g" run_fvt_training.sh > $RUN_FILENAME
		sed -i "s/SIGNAL_RATIO=0.0/SIGNAL_RATIO=${sr}/g" $RUN_FILENAME
		chmod +x $RUN_FILENAME
		sbatch $RUN_FILENAME
		rm -f $RUN_FILENAME
		sleep 0.2
	done
done

