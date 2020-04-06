import shap
shap.initjs()
from data import X_train, Y_train, X_valid, Y_valid
from model import train_xgb_model, train_logistic, simple_network

#xgbmodel = train_xgb_model(X_train, Y_train, X_valid, Y_valid)
nnmodel = simple_network(X_train, Y_train, X_valid, Y_valid)
#logmodel = train_logistic(X_train, Y_train, X_valid, Y_valid)


# plot importance
#explainer = shap.TreeExplainer(xgbmodel)
#xg_shap_values = explainer.shap_values(X_train.iloc[:100, :])
#shap.force_plot(explainer.expected_value, xg_shap_values[0, :], X_train.iloc[0, :], matplotlib=True)
#shap.summary_plot(xg_shap_values, X_train.iloc[:100, :], feature_names=list(X_train.columns))