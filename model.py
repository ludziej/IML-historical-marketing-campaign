from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from ml_utils import local_search
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from pylift.eval import UpliftEval
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def calc_uplift(model, x, treatment_col):
    return predict_treatment(model, treatment_col, 1)(x) - predict_treatment(model, treatment_col, 0)(x)


def predict_treatment(model, treatment_col, c):
    def pred(x):
        x = x.copy()
        x[:, treatment_col] = c
        return model.predict_proba(x)[:, 1]
    return pred


def evaluate_uplift(model, x, y, treatment_col, plot=False):

    x_ones = x.copy()
    x_zeros = x.copy()
    x_ones[:, treatment_col] = 1
    x_zeros[:, treatment_col] = 0

    uplift = (model.predict_proba(x_ones) - model.predict_proba(x_zeros))[:, 1]

    upe = UpliftEval(x[:, treatment_col], y, uplift)
    if plot:
        upe.plot(show_theoretical_max=True, show_practical_max=True, show_no_dogs=True, show_random_selection=True)
        plt.plot()

    return upe.q2_cgains


def get_cv_score(create_model, X, Y, treat_col, cv=6):
    scores = []
    for i in range(cv):
        xt, xv, yt, yv = train_test_split(X, Y, test_size=0.3, stratify=Y)
        model = create_model()
        model.fit(xt, yt)
        scores.append(evaluate_uplift(model, xv, yv, treat_col))
    return np.average(scores)


def local_search_svm(X_train, Y_train, X_valid, Y_valid, treatment_col, just_get_model=False, plot=False):
    init_params = {
        'C': 10.,
        'gamma': 0.1
    }
    limits = {}
    create_model = lambda params: SVC(kernel='rbf', probability=True, **params)
    score_function = lambda params: get_cv_score(lambda: create_model(params), X_train, Y_train, treatment_col, cv=6)
    best_model = create_model(init_params if just_get_model else
                              local_search(init_params, score_function, limits=limits))
    best_model.fit(X_train, Y_train)
    print("train score = {}".format(evaluate_uplift(best_model, X_train, Y_train, treatment_col, plot=plot)))
    print("valid score = {}".format(evaluate_uplift(best_model, X_valid, Y_valid, treatment_col, plot=plot)))
    return best_model


def local_search_xgb(X_train, Y_train, X_valid, Y_valid, treatment_col, just_get_model=False):
    init_params = {'subsample': 0.03375205270351832, 'colsample_bytree': 0.017281050984201383,
                   'learning_rate': 0.007474227529937205, 'min_child_weight': 0.016192759517420326,
                   'gamma': 0.0005306043438668296, 'max_depth': 4, 'n_estimators': 12
                   }
    limits = {'subsample': (0., 1.), 'colsample_bytree': (0., 1.)}
    create_model = lambda params: XGBClassifier(objective='binary:logistic',  **params)
    score_function = lambda params: get_cv_score(lambda: create_model(params), X_train, Y_train, treatment_col, cv=12)
    best_model = create_model(init_params if just_get_model else
                              local_search(init_params, score_function, limits=limits))
    best_model.fit(X_train, Y_train)
    print("train score = {}".format(evaluate_uplift(best_model, X_train, Y_train, treatment_col)))
    print("valid score = {}".format(evaluate_uplift(best_model, X_valid, Y_valid, treatment_col)))
    return best_model


def simple_network(X_train, Y_train, X_valid, Y_valid, channels=100, layers=3, dropout=0.3):

    model = keras.Sequential(
        layers * [
            keras.layers.Dense(channels, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
        ] + [
            keras.layers.Dense(1, activation='sigmoid')
        ]
    )
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall() ,'accuracy', tf.keras.metrics.AUC()])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    model.fit(X_train, Y_train, validation_split=0.3, verbose=2, epochs=2000, callbacks=[es])
    train_acc = model.evaluate(X_train, Y_train, verbose=0)
    print('\nTrain accuracy:', train_acc)
    test_acc = model.evaluate(X_valid, Y_valid, verbose=2)

    print('\nTest accuracy:', test_acc)


def train_xgb_model(X_train, Y_train, X_valid, Y_valid):
    xgmodel = XGBClassifier(max_depth=13,
                            objective='binary:logistic',
                            gamma=0.1)
    xgmodel.fit(X_train, Y_train, verbose=True)
    train_score = xgmodel.score(X_train, Y_train)
    print("XGBoost train score: {}".format(train_score))
    valid_score = xgmodel.score(X_valid, Y_valid)
    print("XGBoost valid score: {}".format(valid_score))
    return xgmodel


def train_logistic(X_train, Y_train, X_valid, Y_valid):
    logmodel = LogisticRegression(solver='liblinear', C=3)
    logmodel.fit(X_train, Y_train)
    train_score = logmodel.score(X_train, Y_train)
    print("Logistic regression train score: {}".format(train_score))
    log_score = logmodel.score(X_valid, Y_valid)
    print("Logistic regression valid score: {}".format(log_score))
    return logmodel
