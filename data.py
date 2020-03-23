import pyreadr

train = pyreadr.read_r('hmc_train.Rda')['train']  # pandas df
valid = pyreadr.read_r('hmc_valid.Rda')['valid']


def xy_split(data, y_name="PURCHASE"):
    return data.drop([y_name], axis=1), data[y_name]


X_train, Y_train = xy_split(train)
X_valid, Y_valid = xy_split(train)
