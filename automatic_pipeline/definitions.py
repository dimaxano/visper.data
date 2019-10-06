from os.path import dirname, realpath, split, join

MODULE_PATH, _ = split(realpath(__file__))
DATA_PATH = join(MODULE_PATH, "data")
RESULTS_PATH = join(MODULE_PATH, "results")