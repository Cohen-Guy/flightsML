import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import KFold

class HyperParametersTunning:
    def __init__(self, train_x, valid_x, train_y, valid_y):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
    def objective(self, trial):
        dtrain = xgb.DMatrix(self.train_cv_X, label=self.train_cv_y)
        dvalid = xgb.DMatrix(self.valid_cv_X, label=self.valid_cv_y)

        param = {
            "verbosity": 0,
            "objective": "multi:softmax",
            "num_class": 11,
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(self.valid_cv_y, pred_labels)
        return accuracy

    def objective_cv(self, trial):
        fold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(self.train_x)))):
            self.train_cv_X = self.train_x.iloc[train_idx]
            self.train_cv_y = self.train_y[train_idx]
            self.valid_cv_X = self.train_x.iloc[valid_idx]
            self.valid_cv_y = self.train_y[valid_idx]
            accuracy = self.objective(trial)
            scores.append(accuracy)
        return np.mean(scores)

    def optimize(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_cv, n_trials=300, timeout=36000)
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
