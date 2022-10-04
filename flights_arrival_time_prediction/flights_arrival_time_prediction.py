import os
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from globalsContext import GlobalsContextClass

class FlightsArrivalTimePrediction:

    def __init__(self):
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.globals_context = GlobalsContextClass(time_str)
        self.debug_flag = True
        dataset_csv_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'flights_arrival_time_prediction', 'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2021_1.csv')
        self.dataset = pd.read_csv(dataset_csv_file_path)
        self.target_bucket_col_name = 'DelayBucket'
        self.categorical_cols = [categorical_col['field_name'] for categorical_col in self.globals_context.cols_dict['categorical_cols']]


    def divide_target_to_buckets(self, dataset, target_col_name, target_bucket_col_name):
        dataset.loc[dataset[target_col_name].between(-10000, -60, 'both'), target_bucket_col_name] = '{-10000, -60}'
        dataset.loc[dataset[target_col_name].between(-60, -45, 'right'), target_bucket_col_name] = '{-60, -45}'
        dataset.loc[dataset[target_col_name].between(-45, -30, 'right'), target_bucket_col_name] = '{-45, -30}'
        dataset.loc[dataset[target_col_name].between(-30, -15, 'right'), target_bucket_col_name] = '{-30, -15}'
        dataset.loc[dataset[target_col_name].between(-15, -5, 'right'), target_bucket_col_name] = '{-15, -5}'
        dataset.loc[dataset[target_col_name].between(-5, 0, 'right'), target_bucket_col_name] = '{-5, 0}'
        dataset.loc[dataset[target_col_name].between(0, 5, 'left'), target_bucket_col_name] = '{0, 5}'
        dataset.loc[dataset[target_col_name].between(5, 15, 'left'), target_bucket_col_name] = '{5, 15}'
        dataset.loc[dataset[target_col_name].between(15, 30, 'left'), target_bucket_col_name] = '{15, 30}'
        dataset.loc[dataset[target_col_name].between(30, 45, 'left'), target_bucket_col_name] = '{30, 45}'
        dataset.loc[dataset[target_col_name].between(45, 60, 'left'), target_bucket_col_name] = '{45, 60}'
        dataset.loc[dataset[target_col_name].between(60, 10000, 'both'), target_bucket_col_name] = '{60, 10000}'
        return dataset

    def dataset_extract_target(self, dataset):
        target_col_name = 'ArrDelay'
        dataset = self.divide_target_to_buckets(dataset, target_col_name, self.target_bucket_col_name)
        y = dataset.pop(self.target_bucket_col_name)
        return dataset, y

    def feature_selection(self, X):
        # self.selected_features = ['origin_location_code', 'destination_location_code', 'departure_date', 'validatingAirlineCodes', 'distance']  # 'validatingAirlineCodes',
        self.excluded_features = ['FlightDate']
        X = X.loc[:, ~X.columns.isin(self.excluded_features)]
        return X

    def feature_engineering(self, X):
        return X

    def outlier_detection(self, X, excluded_column):
        isolation_forest_model = IsolationForest(max_samples=100, random_state=42)
        X_without_excluded_columns = X.loc[:, X.columns != excluded_column]
        X['outlier'] = isolation_forest_model.fit_predict(X_without_excluded_columns)
        return X

    def encoding(self, X, y, categorical_columns):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        target_encoder = TargetEncoder()
        X[categorical_columns] = target_encoder.fit_transform(X[categorical_columns], y)
        return X

    def preprocessing(self, dataset):
        # predict_for_airline_code = 'AF'
        # dataset = self.filter_dataset_by_airline_code(dataset, predict_for_airline_code)
        # dataset = self.cleaning(dataset)
        dataset = self.transform_types(dataset)
        X, y = self.dataset_extract_target(dataset)
        X = self.feature_selection(X)
        # X = self.feature_engineering(X)
        X = self.encoding(X, y, self.categorical_cols)
        # excluded_column = 'departure_date'
        # X = self.outlier_detection(X, excluded_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        return X_train, X_test, y_train, y_test

    def transform_types(self, dataset):
        dataset[self.categorical_cols] = dataset[self.categorical_cols].astype(str)
        return dataset

    def train_linear_regression(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_xgboost_regressor(self, X_train, y_train):
        model = XGBRegressor(colsample_bytree=0.7, learning_rate=0.07, max_depth=7, min_child_weight=4, n_estimators=100)
        model.fit(X_train, y_train)
        return model

    def explainability(self, model, X, y):
        shap_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'explainability', 'shap')
        sample_ind = 20
        sampled_x = X.sample(frac=0.1)
        explainer = shap.Explainer(model, sampled_x)
        shap_values = explainer(sampled_x)
        waterfall_diagram = shap.plots.waterfall(shap_values[0], show=False)
        waterfall_diagram_name = f"waterfall_diagram_{self.time_str}.png"
        waterfall_diagram_path = os.path.join(shap_folder_path, waterfall_diagram_name)
        waterfall_diagram.savefig(waterfall_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()
        shap.plots.beeswarm(shap_values, show=False)
        beeswarm_diagram_name = f"beeswarm_diagram_{self.time_str}.png"
        beeswarm_diagram_path = os.path.join(shap_folder_path, beeswarm_diagram_name)
        beeswarm_diagram = plt.gcf()
        beeswarm_diagram.savefig(beeswarm_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()
        # summary_plot_fig = shap.summary_plot(shap_values[0], X_test, show=False)
        # summary_plot_diagram_name = f"summary_plot_diagram_{self.time_str}.png"
        # summary_plot_path = os.path.join(summary_plot_fig, summary_plot_diagram_name)
        # summary_plot_fig.savefig(summary_plot_path, format='png', dpi=600, bbox_inches='tight')

        # for feature in self.selected_features:
        feature = 'distance'
        self.shap_plots_for_feature(model, feature, shap_values, shap_folder_path, sampled_x, sample_ind)
        # feature = 'departure_date'
        # self.shap_plots_for_feature(feature, shap_values, shap_folder_path, sampled_x, sample_ind)

    def shap_plots_for_feature(self, model, feature, shap_values, shap_folder_path, sampled_x, sample_ind):
        feature_shap_values = shap_values[:, feature]
        shap.plots.scatter(feature_shap_values, color=shap_values, show=False)
        scatter_diagram_name = f"{feature}_scatter_diagram_{self.time_str}.png"
        scatter_diagram_path = os.path.join(shap_folder_path, scatter_diagram_name)
        scatter_diagram = plt.gcf()
        scatter_diagram.savefig(scatter_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()
        partial_dependence_diagram, ax = shap.partial_dependence_plot(
            feature, model.predict, sampled_x, model_expected_value=True,
            feature_expected_value=True, ice=False,
            shap_values=shap_values[sample_ind:sample_ind + 1, :],
            show=False
        )
        partial_dependence_diagram_name = f"{feature}_partial_dependence_diagram_{self.time_str}.png"
        partial_dependence_diagram_path = os.path.join(shap_folder_path, partial_dependence_diagram_name)
        partial_dependence_diagram.savefig(partial_dependence_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()

    def train_random_forest_regressor(self, X_train, y_train):
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        return model

    def evaluate_train(self, model, X, y):
        y_pred = model.predict(X)
        root_mean_square_error = mean_squared_error(y, y_pred, squared=False)
        X['y_true'] = y
        X['y_pred'] = y_pred
        return X, root_mean_square_error

    def evaluate_test(self, model, X, y):
        y_pred = model.predict(X)
        root_mean_square_error = mean_squared_error(y, y_pred, squared=False)
        return X, root_mean_square_error


    def ml_flow(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self.dataset)
        model = self.train_xgboost_regressor(X_train, y_train)
        X_train, train_root_mean_square_error = self.evaluate_train(model, X_train, y_train)
        print(f"train_root_mean_square_error: {train_root_mean_square_error}")
        X_test, test_root_mean_square_error = self.evaluate_test(model, X_test, y_test)
        print(f"test_root_mean_square_error: {test_root_mean_square_error}")
        return model, X_test, y_test

    def hyperparameters_optimization(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self.dataset)
        parameters = {'objective': ['reg:squarederror'],
                      'learning_rate': [.03, 0.05, .07],  # so called `eta` value
                      'max_depth': [5, 6, 7],
                      'min_child_weight': [4],
                      'subsample': [0.7],
                      'colsample_bytree': [0.7],
                      'n_estimators': [1, 4, 10, 50, 100, 500]}

        xgb = XGBRegressor()
        random_search = GridSearchCV(xgb,
                        parameters,
                        cv=5,
                        n_jobs=5,
                        verbose=True)
        random_search.fit(X_train, y_train)
        print(random_search.best_params_)


if __name__ == "__main__":
    flights_arrival_time_prediction = FlightsArrivalTimePrediction()
    # fareMLPrediction.hyperparameters_optimization()
    model, X_test, y_test = flights_arrival_time_prediction.ml_flow()
    # fareMLPrediction.explainability(model, X_test, y_test)