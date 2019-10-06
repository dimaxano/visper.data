from os.path import dirname, realpath, split, join

MODULE_PATH, _ = split(realpath(__file__))
IMG_PATH = join(MODULE_PATH, "data", "crowd.jpg")
PROTOTXT_PATH = join(MODULE_PATH, "model", "deploy.prototxt")
WEIGHTS_PATH = join(MODULE_PATH, "weights")
RESULTS_PATH = join(MODULE_PATH, "results")