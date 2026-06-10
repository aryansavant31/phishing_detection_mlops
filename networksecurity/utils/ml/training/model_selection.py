from networksecurity.exceptions.custom_exception import NetworkSecurityException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import sys
from box import ConfigBox

def get_best_model(X_train, y_train, X_test, y_test, models:dict, hyperparams:ConfigBox):
    try:
        report = {}

        for model_name in models.keys():
            model = models[model_name]
            hyperparam = hyperparams[model_name]

            grid_search = GridSearchCV(model, hyperparam, cv=3)
            grid_search.fit(X_train, y_train)

            # train model with best params
            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            # get test prediction and evalaute
            y_test_pred = model.predict(X_test)
            test_r2_score = r2_score(y_test, y_test_pred)
            report[model_name] = {"score":test_r2_score, "model":model}

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)