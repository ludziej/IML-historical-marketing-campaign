import shap
from data import X_train, Y_train, X_valid, Y_valid, treatment_col, column_names
from model import train_xgb_model, train_logistic, simple_network
from pylift.eval import UpliftEval
import matplotlib

xgbmodel = train_xgb_model(X_train, Y_train, X_valid, Y_valid)

#upev = UpliftEval(X_valid[:, treatment_col], xgbmodel.predict_proba(treatment_col))
#nmodel = simple_network(X_train, Y_train, X_valid, Y_valid)
#logmodel = train_logistic(X_train, Y_train, X_valid, Y_valid)


# plot importance
#explainer = shap.TreeExplainer(xgbmodel)
#xg_shap_values = explainer.shap_values(X_train.iloc[:100, :])
#shap.force_plot(explainer.expected_value, xg_shap_values[0, :], X_train.iloc[0, :], matplotlib=True)
#shap.summary_plot(xg_shap_values, X_train.iloc[:100, :], feature_names=list(X_train.columns))


x_ones = X_valid.copy()
x_zeros = X_valid.copy()
x_ones[:, treatment_col] = 1
x_zeros[:, treatment_col] = 0

uplift = (xgbmodel.predict_proba(x_ones) - xgbmodel.predict_proba(x_zeros))[:, 1]

upe = UpliftEval(X_valid[:, treatment_col], Y_valid, uplift)
upe.plot()
pass