from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from ml_utils import local_search
from sklearn.model_selection import cross_validate
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from utils import flatten
from pylift import eval
from sklearn.model_selection import train_test_split
from pylift.eval import UpliftEval


def evaluate_score(model, x, y, treatment_col, plot=False):

    x_ones = x.copy()
    x_zeros = x.copy()
    x_ones[:, treatment_col] = 1
    x_zeros[:, treatment_col] = 0

    uplift = (model.predict_proba(x_ones) - model.predict_proba(x_zeros))[:, 1]

    upe = UpliftEval(x[:, treatment_col], y, uplift)
    if plot:
        upe.plot(show_theoretical_max=True, show_practical_max=True, show_no_dogs=True, show_random_selection=True)

    return upe.q2_cgains


def get_score(x, y, y_pred, treat_col, plot=False, policy=0.2):
    scores = eval.get_scores(x[:, treat_col], y, y_pred, policy, scoring_range=(0, 1), plot_type='all')
    return scores['overall_lift']


def plot_uplift(model, x, y, treat_col):
    preds = model.predict_proba(x)[:, 1]
    up = eval.UpliftEval(x[:, treat_col], y, preds)
    up.plot()
    score = get_score(x, y, preds, treat_col)
    #print(score)
    return score


def get_cv_score(create_model, X, Y, treat_col, cv=5):
    scores = []
    for i in range(cv):
        xt, xv, yt, yv = train_test_split(X, Y, test_size=0.2, stratify=Y)
        model = create_model()
        model.fit(xt, yt)
        probas = model.predict_proba(xv)[:, 1]
        scores.append(get_score(xv, yv, probas, treat_col))
    return np.average(scores)


def local_search_svm(X_train, Y_train, X_valid, Y_valid):
    pass


def local_search_xgb(X_train, Y_train, X_valid, Y_valid, treatment_col):
    init_params = {
        'subsample': 0.8,
        'colsample_bytree': 1.,
        'learning_rate': 0.1,
        'min_child_weight': 1.,
        'gamma': 0.1,
        'max_depth': 10,
        'n_estimators': 100
    }
    limits = {'subsample': (0., 1.), 'colsample_bytree': (0., 1.)}
    create_model = lambda params: XGBClassifier(objective='binary:logistic',  **params)
#    score_function = lambda params: np.average(cross_validate(create_model(params), X_train, Y_train,
#                                                              cv=5, n_jobs=5, verbose=0)['test_score'])
    score_function = lambda params: get_cv_score(lambda: create_model(params), X_train, Y_train, treatment_col)
    best_params = local_search(init_params, score_function, limits=limits)
    best_model = create_model(best_params)
    best_model.fit(X_train, Y_train,)
    print("train score = {}".format(plot_uplift(best_model, X_train, Y_train, treatment_col)))
    print("valid score = {}".format(plot_uplift(best_model, X_valid, Y_valid, treatment_col)))
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
