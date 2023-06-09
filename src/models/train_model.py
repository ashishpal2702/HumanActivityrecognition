from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)

from src.utils.helper import save_weights, save_model
from sklearn.model_selection import GridSearchCV

# from pycaret.classification import *
from src.models.eval import Eval


class Train:
    def train_and_predict(self, model, x_train, y_train, x_test, y_test):
        e = Eval()
        trained_model = self.train_model(model, x_train, y_train)
        y_test_pred = self.model_predict(trained_model, x_test)
        ## Test data metrics
        score, f1, p, r = e.eval_metrics(y_test_pred, y_test)
        return score, f1, p, r

    def train_model(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        save_model(model)
        return model

    def save_model_weights(self, model, model_path):
        save_weights(model, model_path)

    def model_predict(self, model, x):
        predicted_proba = model.predict_proba(x)
        #y_pred = (predicted_proba[:, 1] >= self.threshold).astype("int")
        y_pred = model.predict(x)
        return y_pred

    def get_classification_models(self):
        lr = LogisticRegression()
        #svc = SVC(probability=True)
        dt = DecisionTreeClassifier()
        ## ensembles
        rfc = RandomForestClassifier()
        bag_clf = BaggingClassifier(
            DecisionTreeClassifier(),
            n_estimators=50,
            max_samples=300,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )

        gbdt_clf = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=1.0,
            max_depth=20,
            max_leaf_nodes=2,
            random_state=42,
        )
        return [lr, dt, rfc, bag_clf, gbdt_clf]

    def get_best_hyperparameters(self, x_train, y_train, param_grid, model):
        CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        CV_rfc.fit(x_train, y_train)
        return CV_rfc.best_params_

    def best_lr_model(self, x, y):
        param_grid = {
            "penalty": ["l1", "l2", "elasticnet"],
            "class_weight": [{0: 1, 1: 2}, {0: 1, 1: 4}],
        }
        rfc = LogisticRegression()
        best_parameters = self.get_best_hyperparameters(x, y, param_grid, rfc)
        rfc.set_params(**best_parameters)
        return rfc

    def best_rfc_model(self, x, y):
        param_grid = {
            "n_estimators": [
                50,
                100,
                200,
            ],
            "max_depth": [4, 12, 20, 40],
            "criterion": ["gini", "entropy"],
            "class_weight": [{0: 1, 1: 2}, {0: 1, 1: 4}],
        }
        rfc = RandomForestClassifier()
        best_parameters = self.get_best_hyperparameters(x, y, param_grid, rfc)
        rfc.set_params(**best_parameters)
        return rfc

    """
    def automl(self,df):
        s = setup(df, target='Churn', ignore_features=['customerID'])
        eval = Eval()
        add_metric('profit', 'Profit', eval.business_matrix())

        best_model = compare_models(sort='Profit')

        tuned_best_model = tune_model(best_model)

        return tuned_best_model
    """
