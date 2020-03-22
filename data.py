import pyreadr

train = pyreadr.read_r('hmc_train.Rda')['train']  # pandas df
valid = pyreadr.read_r('hmc_valid.Rda')['valid']