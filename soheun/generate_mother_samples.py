from dataset import MotherSamples
from itertools import product
from tqdm import tqdm

n_3b = 100_0000
ratio_4b = 0.5
seeds = range(0, 100)
signal_ratios = [0.005, 0.0075, 0.01, 0.02]
signal_filenames = ["ZH4b_picoAOD_cleaned.h5"]

for signal_ratio, signal_filename, seed in tqdm(
    product(signal_ratios, signal_filenames, seeds)
):
    ms_hparams = {
        "n_3b": n_3b,
        "ratio_4b": ratio_4b,
        "signal_ratio": signal_ratio,
        "signal_filename": signal_filename,
        "seed": seed,
    }
    MotherSamples.from_hparams(ms_hparams).save()
