from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def train_xgb_model(X_train, Y_train, X_valid, Y_valid):
    xgmodel = XGBClassifier(max_depth=13,
                            objective='binary:logistic',
                            gamma=0.1)
    xgmodel.fit(X_train, Y_train, verbose=True)
    valid_score = xgmodel.score(X_valid, Y_valid)
    print("xgboost valid score {}".format(valid_score))
    return xgmodel


def train_logistic(X_train, Y_train, X_valid, Y_valid):
    logmodel = LogisticRegression(max_iter=1000, solver='lbfgs', C=3)
    logmodel.fit(X_train, Y_train)
    log_score = logmodel.score(X_valid, Y_valid)
    print("logistic valid score {}".format(log_score))
    return logmodel
