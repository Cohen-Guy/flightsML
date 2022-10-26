import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
class HyperParametersTunning:
    def __init__(self, train_x, valid_x, train_y, valid_y):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y

    def objective_xgb(self, trial):
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

    def objective_cv_xgb(self, trial):
        fold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, valid_idx in fold.split(range(len(self.train_x))):
            self.train_cv_X = self.train_x.iloc[train_idx]
            self.train_cv_y = self.train_y[train_idx]
            self.valid_cv_X = self.train_x.iloc[valid_idx]
            self.valid_cv_y = self.train_y[valid_idx]
            accuracy = self.objective_xgb(trial)
            scores.append(accuracy)
        return np.mean(scores)

    def objective_rf(self, trial):
        params = {
            'n_estimators': trial.suggest_int("rf_n_estimators", 10, 1000),
            'max_depth': trial.suggest_int("rf_max_depth", 2, 32, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
        }

        clf = RandomForestClassifier(random_state=42, **params)
        bst = clf.fit(self.train_cv_X, self.train_cv_y)
        preds = bst.predict(self.valid_cv_X)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(self.valid_cv_y, pred_labels)
        return accuracy

    def objective_cv_rf(self, trial):
        fold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(self.train_x)))):
            self.train_cv_X = self.train_x.iloc[train_idx]
            self.train_cv_y = self.train_y.iloc[train_idx].astype('int')
            self.valid_cv_X = self.train_x.iloc[valid_idx]
            self.valid_cv_y = self.train_y.iloc[valid_idx].astype('int')
            accuracy = self.objective_rf(trial)
            scores.append(accuracy)
        return np.mean(scores)

    def objective_svm(self, trial):
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        clf = SVC(C=svc_c, gamma="auto")
        bst = clf.fit(self.train_cv_X, self.train_cv_y)
        preds = bst.predict(self.valid_cv_X)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(self.valid_cv_y, pred_labels)
        return accuracy

    def objective_cv_svm(self, trial):
        fold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(self.train_x)))):
            self.train_cv_X = self.train_x.iloc[train_idx]
            self.train_cv_y = self.train_y.iloc[train_idx].astype('int')
            self.valid_cv_X = self.train_x.iloc[valid_idx]
            self.valid_cv_y = self.train_y.iloc[valid_idx].astype('int')
            accuracy = self.objective_svm(trial)
            scores.append(accuracy)
        return np.mean(scores)

    def objective_knn(self, trial):
        optimizer = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        rf_max_depth = trial.suggest_int("k_n_neighbors", 2, 10, log=True)
        clf = KNeighborsClassifier(n_neighbors=rf_max_depth, algorithm=optimizer)
        bst = clf.fit(self.train_cv_X, self.train_cv_y)
        preds = bst.predict(self.valid_cv_X)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(self.valid_cv_y, pred_labels)
        return accuracy

    def objective_cv_knn(self, trial):
        fold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(self.train_x)))):
            self.train_cv_X = self.train_x.iloc[train_idx]
            self.train_cv_y = self.train_y.iloc[train_idx].astype('int')
            self.valid_cv_X = self.train_x.iloc[valid_idx]
            self.valid_cv_y = self.train_y.iloc[valid_idx].astype('int')
            accuracy = self.objective_knn(trial)
            scores.append(accuracy)
        return np.mean(scores)

    def objective_nb(self, trial):
        var_smoothing = trial.suggest_float("var_smoothing", 1e-15, 1e-2, log=True)
        clf = GaussianNB(var_smoothing=var_smoothing)
        bst = clf.fit(self.train_cv_X, self.train_cv_y)
        preds = bst.predict(self.valid_cv_X)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(self.valid_cv_y, pred_labels)
        return accuracy

    def objective_cv_nb(self, trial):
        fold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(self.train_x)))):
            self.train_cv_X = self.train_x.iloc[train_idx]
            self.train_cv_y = self.train_y.iloc[train_idx].astype('int')
            self.valid_cv_X = self.train_x.iloc[valid_idx]
            self.valid_cv_y = self.train_y.iloc[valid_idx].astype('int')
            accuracy = self.objective_nb(trial)
            scores.append(accuracy)
        return np.mean(scores)

    def optimize(self):
        study = optuna.create_study(direction="maximize")
        # print('RandomForestClassifier')
        # study.optimize(self.objective_cv_rf, n_trials=600, timeout=100000)
        print('SVC')
        study.optimize(self.objective_cv_svm, n_trials=600, timeout=100000)
        # print('KNeighborsClassifier')
        # study.optimize(self.objective_cv_knn, n_trials=600, timeout=100000)
        # print('GaussianNB')
        # study.optimize(self.objective_cv_nb, n_trials=600, timeout=100000)
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
