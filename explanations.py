import shap
from model import predict_treatment


def shapley_tree(model_predict, dataset, obs, column_names, plot_draw=False):
    explainer = shap.KernelExplainer(model_predict, shap.sample(dataset, 100))
    shap_values = explainer.shap_values(obs)
    if plot_draw:
        plot = shap.waterfall_plot(explainer.expected_value, shap_values,
                                       feature_names=column_names)
    return shap_values, explainer.expected_value


def shapley_diff(model, obs, dataset, column_names, treatment_col):
    shap_t0, exp0 = shapley_tree(predict_treatment(model, treatment_col, 0), dataset, obs, column_names)
    shap_t1, exp1 = shapley_tree(predict_treatment(model, treatment_col, 1), dataset, obs, column_names)
    plot = shap.waterfall_plot(exp1 - exp0, shap_t1 - shap_t0, feature_names=column_names)
    return shap_t1 - shap_t0, exp1 - exp0
