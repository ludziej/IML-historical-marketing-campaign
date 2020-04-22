import shap
shap.initjs()
from data import X_train, Y_train, X_valid, Y_valid, treatment_col
from model import train_xgb_model, train_logistic, local_search_xgb


best_model = local_search_xgb(X_train, Y_train, X_valid, Y_valid, treatment_col)
best_model = local_search_xgb(X_train, Y_train, X_valid, Y_valid, treatment_col)
pass
#best_model = seach_xgb_parameters(X_train, Y_train, X_valid, Y_valid)

