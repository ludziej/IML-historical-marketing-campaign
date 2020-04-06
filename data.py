import pyreadr


train = pyreadr.read_r('hmc_train.Rda')['train']  # pandas df
valid = pyreadr.read_r('hmc_valid.Rda')['valid']

train = train[sorted(train.columns)]
valid = valid[sorted(valid.columns)]

assert set(train.columns) == set(valid.columns)

def xy_split(data, y_name="PURCHASE"):
    return data.drop([y_name], axis=1).to_numpy(), data[y_name].to_numpy()


X_train, Y_train = xy_split(train)
X_valid, Y_valid = xy_split(valid)
pass